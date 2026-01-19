import serial
import serial.tools.list_ports
import time
import os

# --- KONFIGURATION ---
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
DATEI_NAME = 'messdaten.csv'
SAMPLES_PER_RECORD = 100
MAX_RECORDINGS_PER_SYMBOL = 10
MAX_SYMBOL_INDEX = 26  # Z
# ---------------------


def list_available_ports():
    """Listet alle verfügbaren COM-Ports auf."""
    ports = serial.tools.list_ports.comports()
    print("Verfügbare Ports:")
    found = False
    for p in ports:
        print(f"  - {p.device} ({p.description})")
        if p.device == SERIAL_PORT:
            found = True
    if not found:
        print(f"\nWARNUNG: Der konfigurierte Port {SERIAL_PORT} wurde NICHT gefunden!")
        if ports:
            print(f"Tipp: Ändere SERIAL_PORT im Skript auf '{ports[0].device}'.")
    print("-" * 30)


def get_last_state(filename):
    """Ermittelt den korrekten Startzustand basierend auf der CSV."""
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return 1, 1  # Start bei A, ID 1

    last_symbol = 1
    last_id = 0

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Rückwärts suchen nach der letzten gültigen Datenzeile
            for i in range(len(lines) - 1, 0, -1):
                line = lines[i].strip()
                if not line:
                    continue
                parts = line.split(';')
                if len(parts) >= 2:
                    try:
                        last_symbol = int(parts[0])
                        last_id = int(parts[1])
                        break
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Warnung: Fehler beim Lesen der CSV ({e}).")

    # Logik: Welches Symbol machen wir als nächstes?
    target_symbol = last_symbol

    # Wenn das letzte Symbol voll war (>= 10), wechseln wir zum nächsten
    # Aber nur, wenn wir nicht schon beim Maximum (Z) sind
    if last_id >= MAX_RECORDINGS_PER_SYMBOL and last_symbol < MAX_SYMBOL_INDEX:
        target_symbol += 1

    # Jetzt schauen wir, was die nächste freie ID für dieses Ziel-Symbol ist
    # Das verhindert, dass wir bei 1 anfangen, wenn für das neue Symbol schon Daten existieren
    next_id = get_next_id_for_symbol(filename, target_symbol)

    return target_symbol, next_id


def send_status_to_esp(ser, symbol_int, count):
    """Sendet STATUS:SYMBOL_CHAR:COUNT an den ESP."""
    try:
        symbol_char = chr(ord('A') + symbol_int - 1)
        msg = f"STATUS:{symbol_char}:{count}\n"
        ser.write(msg.encode('utf-8'))
    except Exception as e:
        print(f"Fehler beim Senden an ESP: {e}")
        raise e


def get_next_id_for_symbol(filename, symbol_val):
    """Sucht in der CSV nach der höchsten ID für ein bestimmtes Symbol."""
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return 1

    max_id = 0
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # Überspringe Header
            next(f, None)
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 2:
                    try:
                        sym = int(parts[0])
                        if sym == symbol_val:
                            rid = int(parts[1])
                            if rid > max_id:
                                max_id = rid
                    except ValueError:
                        continue
    except Exception:
        pass

    return max_id + 1


def main():
    print("--- PC-Logger V2 gestartet ---")
    list_available_ports()
    print(f"Versuche Port {SERIAL_PORT} zu öffnen...")

    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)

        # Reset ESP
        ser.setDTR(False)
        ser.setRTS(False)
        time.sleep(0.1)
        ser.setDTR(True)
        ser.setRTS(True)
        time.sleep(2)

        # Initialen Status ermitteln
        current_symbol_val, current_id = get_last_state(DATEI_NAME)
        print(f"Status: Symbol={chr(ord('A') + current_symbol_val - 1)}, Nächste ID={current_id}")

        # CSV Header check
        file_exists = os.path.exists(DATEI_NAME) and os.path.getsize(DATEI_NAME) > 0
        if not file_exists:
            with open(DATEI_NAME, 'w', encoding='utf-8', newline='') as f:
                ms_labels = [(f"{(i+1)/50:.2f}".replace('.', ',') + "s") for i in range(SAMPLES_PER_RECORD)]
                header = "symbol;aufnahme_id;" + ";".join(ms_labels) + "\n"
                f.write(header)

        # Status sofort senden
        try:
            send_status_to_esp(ser, current_symbol_val, current_id - 1)
        except Exception:
            print("Kritischer Fehler: Konnte ESP nach Reset nicht erreichen.")
            return

        buffer = ""
        current_samples = []
        is_receiving = False
        last_status_time = 0

        print("System bereit. Warte auf Daten... (Drücke Strg+C zum Beenden)")

        while True:
            # 1. Status regelmäßig senden (Heartbeat)
            if time.time() - last_status_time > 2.0:
                try:
                    send_status_to_esp(ser, current_symbol_val, current_id - 1)
                    last_status_time = time.time()
                except Exception:
                    print("Verbindung zum ESP verloren!")
                    break

            # 2. Daten lesen
            try:
                if ser.in_waiting:
                    chunk = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += chunk
            except Exception as e:
                # Fehler beim Lesen ignorieren, aber kurz warten um CPU nicht zu grillen
                time.sleep(0.1)
                continue

            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue

                if line == "CMD:NEXT_SYMBOL":
                    if current_symbol_val < MAX_SYMBOL_INDEX:
                        current_symbol_val += 1
                        current_id = get_next_id_for_symbol(DATEI_NAME, current_symbol_val)
                        print(f"-> Wechsel zu Symbol: {chr(ord('A') + current_symbol_val - 1)} (Nächste ID: {current_id})")
                        send_status_to_esp(ser, current_symbol_val, current_id - 1)
                    else:
                        print("-> Maximales Symbol erreicht.")

                elif line == "CMD:PREV_SYMBOL":
                    if current_symbol_val > 1:
                        current_symbol_val -= 1
                        current_id = get_next_id_for_symbol(DATEI_NAME, current_symbol_val)
                        print(f"-> Wechsel zu Symbol: {chr(ord('A') + current_symbol_val - 1)} (Nächste ID: {current_id})")
                        send_status_to_esp(ser, current_symbol_val, current_id - 1)
                    else:
                        print("-> Minimales Symbol erreicht.")

                elif line == "START_RECORD":
                    print(f"Aufnahme gestartet... (Symbol {chr(ord('A') + current_symbol_val - 1)})")
                    current_samples = []
                    is_receiving = True

                elif line == "END_RECORD":
                    if is_receiving:
                        # Speichern
                        if len(current_samples) < SAMPLES_PER_RECORD:
                            current_samples.extend(['0,0'] * (SAMPLES_PER_RECORD - len(current_samples)))
                        elif len(current_samples) > SAMPLES_PER_RECORD:
                            current_samples = current_samples[:SAMPLES_PER_RECORD]

                        with open(DATEI_NAME, 'a', encoding='utf-8', newline='') as f:
                            row = [str(current_symbol_val), str(current_id)] + current_samples
                            f.write(";".join(row) + "\n")
                            f.flush()
                            os.fsync(f.fileno())

                        print(f"Gespeichert: {chr(ord('A') + current_symbol_val - 1)} - ID {current_id}")

                        # Logik für nächstes Symbol/ID
                        current_id += 1
                        # Hier KEIN automatischer Wechsel mehr, da wir Taster haben!
                        # Wir bleiben beim aktuellen Symbol, bis der User wechselt.

                        # Sofort neuen Status senden
                        send_status_to_esp(ser, current_symbol_val, current_id - 1)
                        is_receiving = False

                elif is_receiving:
                    try:
                        val = float(line)
                        current_samples.append(f"{val:.4f}".replace('.', ','))
                    except ValueError:
                        pass

                elif line.startswith("DEBUG"):
                    print(f"[ESP] {line}")

            # Kurze Pause um CPU zu schonen
            time.sleep(0.01)

    except serial.SerialException as e:
        print(f"\nSerieller Fehler: {e}")
        print("Bitte ESP32 neu anstecken und Skript neu starten.")
    except KeyboardInterrupt:
        print("\nBeendet.")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("Port geschlossen.")


if __name__ == '__main__':
    main()
