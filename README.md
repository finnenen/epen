# E-Pen Data Collector & Gesture Recognition V2

Dieses Projekt ist ein vollständiges System zur Erfassung, Analyse und Klassifizierung von Sensordaten eines "E-Pen" mit einem ESP32. Es ermöglicht das Sammeln von Trainingsdaten, das Trainieren eines neuronalen Netzes (ResNet-ähnlich) und die interaktive Vorhersage von Gesten/Symbolen in Echtzeit.

## Projektstruktur

- **ESP32 Firmware**:
    - `esp32_main.py`: Hauptlogik für den ESP32 (Datenerfassung, LCD-Steuerung, Serielle Kommunikation).
    - `i2c_lcd.py`, `lcd_api.py`: Treiber für das I2C LCD-Display.

- **PC Tools**:
    - `data-collector.py`: Empfängt Sensordaten vom ESP32 über Serial und speichert sie in einer CSV-Datei.
    - `training-model.py`: Trainiert ein Keras/TensorFlow-Modell basierend auf den gesammelten Daten.
    - `predict_interactive.py`: Lädt ein trainiertes Modell und klassifiziert eingehende Daten vom ESP32 in Echtzeit.

## Voraussetzungen

- **Python 3.8+** auf dem PC
- **MicroPython** auf dem ESP32 installiert

### Abhängigkeiten installieren

Es wird empfohlen, eine virtuelle Umgebung zu erstellen:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

Installiere die notwendigen Pakete:

```bash
pip install -r requirements.txt
```

## Hardware Setup (ESP32)

Die Pin-Belegung ist in `esp32_main.py` konfiguriert:
- **ADC Input**: GPIO 34 (Dämpfung 11dB für vollen Bereich)
- **Status LED**: GPIO 2 (Interne LED)
- **Externe LED**: GPIO 23
- **Buttons**:
    - Previous: GPIO 18
    - Next: GPIO 19
- **I2C LCD**:
    - SDA: GPIO 21
    - SCL: GPIO 22

Stelle sicher, dass die Dateien `esp32_main.py` (als `main.py` empfohlen), `i2c_lcd.py` und `lcd_api.py` auf den ESP32 hochgeladen sind.

## Verwendung

### 1. Daten Sammeln
Starten Sie das Skript, um Daten vom ESP32 aufzuzeichnen und in `messdaten.csv` zu speichern.
Stellen Sie sicher, dass der korrekte COM-Port im Skript (`SERIAL_PORT`) eingestellt ist.

```bash
python data-collector.py
```
*Folgen Sie den Anweisungen auf dem LCD-Display des ESP32, um verschiedene Symbole/Gesten aufzuzeichnen.*

### 2. Modell Trainieren
Nachdem genügend Daten gesammelt wurden, kann das Modell trainiert werden.

```bash
# Training starten
python training-model.py train
```
Das trainierte Modell wird standardmäßig im Ordner `models/` gespeichert.

### 3. Interaktive Vorhersage
Starten Sie den interaktiven Modus, um Gesten in Echtzeit zu erkennen. Das Skript verbindet sich mit dem ESP32 und sendet Klassifikationsergebnisse zurück auf das LCD.

```bash
python predict_interactive.py
```
*Wählen Sie beim Start das gewünschte Modell aus dem `models/` Ordner aus.*
