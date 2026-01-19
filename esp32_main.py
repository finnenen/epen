import machine
import time
import sys
import select
from machine import SoftI2C, Pin
from i2c_lcd import I2cLcd

# --- KONFIGURATION ---
ADC_PIN = 34
LED_PIN = 2
EXT_LED_PIN = 23
BTN_PREV_PIN = 18
BTN_NEXT_PIN = 19
THRESHOLD = 0.2
RECORD_DURATION = 2.0
BUCKET_MS = 20  # 50 Hz = 20ms
NUM_SAMPLES = int(RECORD_DURATION * 1000 / BUCKET_MS)

# LCD Konfiguration
I2C_ADDR = 0x27 # Häufige Adresse, kann auch 0x3F sein
I2C_NUM_ROWS = 2
I2C_NUM_COLS = 16

adc = machine.ADC(machine.Pin(ADC_PIN))
adc.atten(machine.ADC.ATTN_11DB)
adc.width(machine.ADC.WIDTH_12BIT)

led = machine.Pin(LED_PIN, machine.Pin.OUT)
ext_led = machine.Pin(EXT_LED_PIN, machine.Pin.OUT)
btn_prev = machine.Pin(BTN_PREV_PIN, machine.Pin.IN)
btn_next = machine.Pin(BTN_NEXT_PIN, machine.Pin.IN)

# I2C und LCD initialisieren
try:
    i2c = SoftI2C(scl=Pin(22), sda=Pin(21), freq=100000)
    # Scan nach Geräten (optional, hilft bei falscher Adresse)
    devices = i2c.scan()
    if len(devices) > 0:
        I2C_ADDR = devices[0] # Nimm das erste gefundene Gerät
    lcd = I2cLcd(i2c, I2C_ADDR, I2C_NUM_ROWS, I2C_NUM_COLS)
    lcd.clear()
except Exception as e:
    print(f"LCD Fehler: {e}")
    lcd = None

# Globale Variablen für Status
pc_connected = False
current_symbol = '?'
current_count = 0
last_result_text = None  # Speichert das letzte Ergebnis (für Predict Modus)
last_msg_time = 0
TIMEOUT_SEC = 2.5

# Entprellung
last_btn_time = 0
BTN_DEBOUNCE_MS = 300

# Animation
anim_frame = 0
last_anim_time = 0

def read_serial_input():
    """Prüft nicht-blockierend auf Nachrichten vom PC."""
    global pc_connected, current_symbol, current_count, last_msg_time, last_result_text
    
    try:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            
            # 1. STATUS Nachricht (Heartbeat)
            if line.startswith("STATUS:"):
                # STATUS:{SymbolStr}:{CountInt}
                # Für Predict Modus sendet PC: STATUS:INTERPRET:0
                parts = line.split(':')
                current_symbol = parts[1]
                current_count = int(parts[2])
                pc_connected = True
                last_msg_time = time.time()
                update_display_connected()
            
            # 2. RESULT Nachricht (Nach Prediction)
            # RESULT:{Label}:{Prob}
            elif line.startswith("RESULT:"):
                parts = line.split(':')
                if len(parts) >= 3:
                    label = parts[1]
                    prob = parts[2]
                    # Speichere Ergebnis für Anzeige
                    last_result_text = f"{label} ({prob}%)"
                    update_display_connected()
                    
    except Exception:
        pass

def update_display_waiting():
    global anim_frame, last_anim_time
    if lcd is None: return
    
    now = time.time()
    if now - last_anim_time > 0.5: # Alle 0.5s aktualisieren
        lcd.move_to(0, 0)
        lcd.putstr("Programm starten")
        
        lcd.move_to(0, 1)
        dots = "." * ((anim_frame % 3) + 1)
        lcd.putstr(f"{dots:<16}") # Links ausrichten, Rest mit Leerzeichen füllen
        
        anim_frame += 1
        last_anim_time = now

def update_display_connected():
    if lcd is None: return
    
    if current_count == -1:
        # --- PREDICT / INTERPRET MODUS ---
        # Zeile 1: Modellname (zentriert in ><)
        text = f"<{current_symbol}>"
        padding = (16 - len(text)) // 2
        # Begrenzen auf 16 Zeichen, falls Name zu lang
        if len(text) > 16: text = text[:16]
        
        lcd.move_to(0, 0)
        lcd.putstr(" " * 16)
        lcd.move_to(max(0, padding), 0)
        lcd.putstr(text)
        
        # Zeile 2: Ergebnis
        lcd.move_to(0, 1)
        if last_result_text:
            lcd.putstr(f"{last_result_text:<16}")
        else:
            lcd.putstr("Ready...        ")
            
    else:
        # --- DATEN SAMMEL MODUS (Data-Collector) ---
        # Zeile 1: Zentriert das Symbol
        text = f">{current_symbol}<"
        padding = (16 - len(text)) // 2
        lcd.move_to(0, 0)
        lcd.putstr(" " * 16) # Zeile leeren
        lcd.move_to(padding, 0)
        lcd.putstr(text)
        
        # Zeile 2: Rec links, Bar rechts (4 Zeichen)
        lcd.move_to(0, 1)
        lcd.putstr(f"Rec:{current_count:<3}")
        
        # Bar Reset (Pos 12 bis 15)
        lcd.move_to(12, 1)
        lcd.putstr("____")

def update_display_recording(elapsed_sec):
    if lcd is None: return
    
    # Progress berechnen (0 bis 4 Zellen)
    progress = elapsed_sec / RECORD_DURATION
    if progress > 1.0: progress = 1.0
    
    filled = int(progress * 4)
    # Bar erstellen: z.B. "##__"
    bar = "#" * filled + "_" * (4 - filled)
    
    # Im Interpret-Modus ist die Bar an einer anderen Stelle oder gleich?
    # User sagt: "unten die selbe progresbar animation"
    # Da "Warte auf Input " oder "Rec:..." verwendet wird, setzen wir die 
    # Bar einfach IMMER nach rechts unten (Pos 12).
    # Das überschreibt ggf. das Ergebnis kurzzeitig, was okay ist.
    
    lcd.move_to(12, 1)
    lcd.putstr(bar)

def blink_waiting():
    led.value(1)
    ext_led.value(1)
    update_display_waiting()
    time.sleep(0.1)
    led.value(0)
    ext_led.value(0)
    time.sleep(0.4) # Etwas schnellerer Loop für flüssigere Animation

def main():
    global pc_connected, last_btn_time
    
    led.value(0)
    print("ESP32 gestartet. Warte auf PC...")
    if lcd:
        lcd.clear()

    while True:
        read_serial_input()

        if pc_connected and (time.time() - last_msg_time > TIMEOUT_SEC):
            pc_connected = False
            current_symbol = '?'
            if lcd: lcd.clear()

        if not pc_connected:
            blink_waiting()
            continue
        
        led.value(0)
        ext_led.value(0)

        # Taster abfragen
        now = time.ticks_ms()
        if time.ticks_diff(now, last_btn_time) > BTN_DEBOUNCE_MS:
            if btn_prev.value() == 1:
                print("CMD:PREV_SYMBOL")
                last_btn_time = now
            elif btn_next.value() == 1:
                print("CMD:NEXT_SYMBOL")
                last_btn_time = now

        # Messen
        val_raw = adc.read()
        val_norm = val_raw / 4095.0

        if val_norm > THRESHOLD:
            led.value(1)
            ext_led.value(1)
            print("START_RECORD")
            
            start_time = time.ticks_ms()
            try:
                for i in range(NUM_SAMPLES):
                    loop_start = time.ticks_ms()
                    sample = adc.read() / 4095.0
                    print(f"{sample:.4f}")
                    
                    # Display Update (nicht bei jedem Sample, sonst zu langsam)
                    # Alle 100ms (jedes 5. Sample bei 20ms)
                    if i % 5 == 0:
                        elapsed = time.ticks_diff(loop_start, start_time) / 1000.0
                        update_display_recording(elapsed)

                    elapsed = time.ticks_diff(time.ticks_ms(), loop_start)
                    sleep_ms = BUCKET_MS - elapsed
                    if sleep_ms > 0:
                        time.sleep_ms(sleep_ms)
                
                # Am Ende volle Zeit anzeigen
                update_display_recording(RECORD_DURATION)
                print("END_RECORD")
            except Exception as e:
                print(f"ERROR: {e}")
            
            # LEDs anlassen während der busy-time/Speichern (verkürzt auf 0.2s)
            time.sleep(0.2)
            
            led.value(0)
            ext_led.value(0)
            
            # Zurücksetzen auf 0.00s
            update_display_connected()
        
        time.sleep(0.02)

if __name__ == '__main__':
    main()
