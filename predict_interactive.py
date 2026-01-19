"""
predict_interactive.py

Dieses Skript:
1. Lädt das trainierte Modell.
2. Verbindet sich mit dem ESP32.
3. Sendet "STATUS:INTERPRET:0", damit der ESP in den Interpretations-Modus schaltet.
4. Empfängt Aufnahmen (START_RECORD ... END_RECORD) wie der Data-Collector.
5. Klassifiziert die Aufnahme.
6. Sendet das Ergebnis (RESULT:{Label}:{Prob}) zurück an den ESP.
"""
import time
import sys
import serial
import serial.tools.list_ports
import os
import joblib
import numpy as np
import tensorflow as tf

# --- KONFIGURATION ---
BAUD_RATE = 115200
MODELS_ROOT = 'models' 

# --- GLOBALS ---
ser = None

def get_available_models():
    """Listet alle Unterordner in models/ auf."""
    if not os.path.exists(MODELS_ROOT):
        return []
    return [d for d in os.listdir(MODELS_ROOT) if os.path.isdir(os.path.join(MODELS_ROOT, d))]

def find_esp32_port():
    """Sucht automatisch nach einem ESP32 Port (CP210x oder CH340)."""
    ports = list(serial.tools.list_ports.comports())
    # 1. Suche nach bekannten Treibern
    for p in ports:
        desc = p.description.lower()
        if "cp210x" in desc or "usb to uart" in desc or "ch340" in desc:
            return p.device
    
    # 2. Fallback: Nimm COM3 wenn vorhanden (User nutzt COM3 im anderen Script)
    for p in ports:
        if p.device.upper() == "COM3":
            return "COM3"
            
    # 3. Wenn nur EIN Port da ist (außer Standard COM1/2 oft), nimm den
    real_ports = [p for p in ports if "Kommunikationsanschluss" not in p.description]
    if len(real_ports) == 1:
        return real_ports[0].device

    return None

def load_model_and_artifacts(model_name):
    model_dir = os.path.join(MODELS_ROOT, model_name)
    print(f"Lade Modell aus {model_dir}...")
    try:
        # 1. Scaler
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        scaler = joblib.load(scaler_path)
        
        # 2. Label Encoder
        le_path = os.path.join(model_dir, 'label_encoder.joblib')
        le = joblib.load(le_path)
        
        # 3. Keras Modell
        model_path = os.path.join(model_dir, 'best_model.h5')
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, 'final_model.h5')
        
        model = tf.keras.models.load_model(model_path)
        
        print(f"Modell '{model_name}' erfolgreich geladen.")
        return model, scaler, le
    except Exception as e:
        print(f"Fehler beim Laden von '{model_name}': {e}")
        return None, None, None

def predict_single_sample(model, scaler, le, raw_values):
    """Nimmt eine Liste von Floats, skaliert sie und gibt Prediction zurück."""
    
    # Keras erwartet numpy array shape (1, Timesteps)
    series = np.array(raw_values, dtype=float).reshape(1, -1)
    
    # Skalieren (erwartet 2D array [n_samples, n_features])
    # Achtung: Scaler wurde auf Trainingsdaten gefittet (Timesteps als Features)
    # Prüfen ob Länge passt:
    expected_len = scaler.n_features_in_
    current_len = series.shape[1]
    
    if current_len != expected_len:
        # Simple Resampling / Padding falls Länge nicht passt (z.B. Timings anders)
        print(f"Warnung: Input Länge {current_len} != Modell Länge {expected_len}. Passe an...")
        series = np.interp(
            np.linspace(0, 1, expected_len),
            np.linspace(0, 1, current_len),
            series[0]
        ).reshape(1, -1)
    
    series_scaled = scaler.transform(series)
    
    # Reshape für CNN (Batch, Steps, Channels) -> (1, 100, 1)
    series_cnn = series_scaled[..., np.newaxis]
    
    # Predict
    probs = model.predict(series_cnn, verbose=0)[0]
    idx = np.argmax(probs)
    prob = probs[idx]
    label = le.inverse_transform([idx])[0]
    
    return label, prob

def get_symbol_char(label):
    """Konvertiert numerische Labels (1='A') in Buchstaben, falls möglich."""
    try:
        val = int(label)
        if 1 <= val <= 26:
            return chr(ord('A') + val - 1)
    except:
        pass
    # Fallback: Original Label nehmen
    return str(label)

def main():
    global ser
    
    # Modelle scannen
    available_models = get_available_models()
    if not available_models:
        print("Keine Modelle gefunden! Bitte trainiere erst eins (training-model.py).")
        return
        
    print("Verfügbare Modelle:", available_models)
    
    # Startmodell wählen (H_O_model wenn da, sonst erstes)
    current_model_idx = 0
    if "H_O_model" in available_models:
        current_model_idx = available_models.index("H_O_model")
    
    current_model_name = available_models[current_model_idx]
    
    # 1. Initiales Modell laden
    model, scaler, le = load_model_and_artifacts(current_model_name)
    if not model:
        return

    # 2. Port finden
    port_name = find_esp32_port()
    if not port_name:
        print("Kein ESP32 gefunden! Bitte anschließen.")
        return
    
    try:
        ser = serial.Serial(port_name, BAUD_RATE, timeout=0.1)
        print(f"Verbunden mit {port_name}")
    except Exception as e:
        print(f"Fehler beim Öffnen von {port_name}: {e}")
        return

    # 3. Hauptschleife
    print("Modus: INTERPRETATION. Warte auf Daten vom ESP...")
    
    last_status_time = 0
    buffer = []
    recording = False
    
    try:
        while True:
            # Heartbeat senden (alle 1s)
            # Sendet STATUS:{ModelName}:-1
            # -1 signalisiert dem ESP den Predict Mode
            if time.time() - last_status_time > 1.0:
                try:
                    msg = f"STATUS:{current_model_name}:-1\n"
                    ser.write(msg.encode('utf-8'))
                    last_status_time = time.time()
                except serial.SerialException:
                    print("Verbindung verloren.")
                    break

            # Daten lesen
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if line == "START_RECORD":
                    print("\n--- Aufnahme gestartet ---")
                    buffer = []
                    recording = True
                
                elif line == "END_RECORD":
                    if recording:
                        print(f"Aufnahme beendet. {len(buffer)} Samples empfangen.")
                        recording = False
                        
                        if len(buffer) > 10: 
                            # PREDICT
                            raw_label, prob = predict_single_sample(model, scaler, le, buffer)
                            prob_percent = prob * 100
                            label_char = get_symbol_char(raw_label)
                            
                            print(f">>> Ergebnis: {label_char} ({prob_percent:.1f}%) [Raw: {raw_label}]")
                            
                            # Rückmeldung an ESP senden
                            msg = f"RESULT:{label_char}:{prob_percent:.1f}\n"
                            ser.write(msg.encode('utf-8'))
                        else:
                            print("Zu wenige Daten.")
                
                elif line == "CMD:NEXT_SYMBOL":
                    # Modell wechseln (vorwärts)
                    current_model_idx = (current_model_idx + 1) % len(available_models)
                    current_model_name = available_models[current_model_idx]
                    print(f"\n>>> Wechsele zu Modell: {current_model_name}")
                    model, scaler, le = load_model_and_artifacts(current_model_name)
                    # Sende sofort Update
                    msg = f"STATUS:{current_model_name}:-1\n"
                    ser.write(msg.encode('utf-8'))

                elif line == "CMD:PREV_SYMBOL":
                    # Modell wechseln (rückwärts)
                    current_model_idx = (current_model_idx - 1) % len(available_models)
                    current_model_name = available_models[current_model_idx]
                    print(f"\n>>> Wechsele zu Modell: {current_model_name}")
                    model, scaler, le = load_model_and_artifacts(current_model_name)
                    msg = f"STATUS:{current_model_name}:-1\n"
                    ser.write(msg.encode('utf-8'))
                            
                elif recording:
                    # Sammle Floats
                    try:
                        val = float(line)
                        buffer.append(val)
                    except ValueError:
                        pass 
                    
            time.sleep(0.005)
            
    except KeyboardInterrupt:
        print("Beende...")
    finally:
        if ser and ser.is_open:
            ser.close()

if __name__ == '__main__':
    main()
