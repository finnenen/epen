import machine
import time

# ADC auf Pin 34 konfigurieren
adc = machine.ADC(machine.Pin(34))
adc.atten(machine.ADC.ATTN_11DB)   # voller Eingangs-Spannungsbereich (~0-3.3..3.6V)
adc.width(machine.ADC.WIDTH_12BIT)  # 0-4095

# Sampling: 50 Hz, also 20 ms Interval
SAMPLE_HZ = 50
SAMPLE_INTERVAL_MS = int(1000 / SAMPLE_HZ)

# Kurzes Start-Delay, damit der serielle Monitor/PC bereit ist
time.sleep_ms(1000)

# Startzeit für Zeitstempel
t0 = time.ticks_ms()

MAX_WERT = 4095.0

# Endlosschleife: Ausgabe im CSV-ähnlichen Format: "elapsed_ms,normalized_value"
# Wichtig: Keine anderen print()-Ausgaben, damit der PC-Logger die Zeilen zuverlässig parst.
while True:
    raw = adc.read()  # 0..4095
    elapsed_ms = time.ticks_diff(time.ticks_ms(), t0)
    # Normalisiere auf 0..1 (float)
    wert_norm = raw / MAX_WERT
    # Ausgabe: timestamp(ms),normalisierterWert
    # Beispiel: 12345,0.5123
    try:
        print("%d,%f" % (elapsed_ms, wert_norm))
    except Exception:
        # sehr seltener Fallback
        print(f"{elapsed_ms},{wert_norm}")
    # warte sample interval
    time.sleep_ms(SAMPLE_INTERVAL_MS)
