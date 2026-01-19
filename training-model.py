"""
training-model.py

Ein einfaches Keras-basiertes Zeitreihen-Klassifikationsskript.
- Liest `messdaten.csv` (Semikolon getrennt, Dezimaltrennzeichen ist Komma)
- Baut ein robustes 1D-ResNet-ähnliches Modell
- Trainings-/Eval-/Predict-Funktionen

Benutzung (Kurz):
    python training-model.py train
    python training-model.py predict --input sample.npy

Hinweis: Installiere Abhängigkeiten aus requirements.txt (tensorflow, pandas, scikit-learn,...)
"""
import sys
from typing import Tuple
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def load_data(csv_path: str, allowed_symbols: list = None) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Lade CSV, konvertiere Dezimal-Komma und gib X,y sowie LabelEncoder zurück.

    Erwartet Spalten: `symbol`, `aufnahme_id`, dann Zeitstempel-Spalten.
    """
    df = pd.read_csv(csv_path, sep=';', decimal=',')

    # Prüfe Mindestanforderungen
    if 'symbol' not in df.columns:
        raise ValueError("CSV muss eine Spalte 'symbol' enthalten")

    # Filter nach Symbolen falls gewünscht
    if allowed_symbols:
        # Sicherstellen dass Vergleich funktioniert (alles als String behandeln)
        df['symbol'] = df['symbol'].astype(str)
        df = df[df['symbol'].isin(allowed_symbols)]
        if df.empty:
            raise ValueError(f"Keine Daten für die Symbole {allowed_symbols} gefunden!")

    # Labels
    y_raw = df['symbol'].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Features: alle anderen Spalten außer symbol und aufnahme_id
    Xdf = df.drop(columns=[c for c in ['symbol', 'aufnahme_id'] if c in df.columns])

    # Falls noch nicht float (z. B. Komma-Dezimal wurde bereits gehandhabt durch read_csv)
    X = Xdf.astype(float).values

    return X, y, le


def preprocess(X: np.ndarray, scaler: StandardScaler = None) -> Tuple[np.ndarray, StandardScaler]:
    """Skaliere Features per Zeit-Feature (fit auf Trainingsdaten)."""
    # X shape: (n_samples, n_timesteps)
    if scaler is None:
        scaler = StandardScaler()
        X2 = scaler.fit_transform(X)
    else:
        X2 = scaler.transform(X)

    # Keras erwartet (samples, timesteps, channels)
    X3 = X2[..., np.newaxis]
    return X3, scaler


def residual_block(x, filters, kernel_size=8):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Falls Shortcut andere Filterzahl hat
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([shortcut, x])
    x = layers.ReLU()(x)
    return x


def build_model(input_shape: Tuple[int, int], n_classes: int) -> models.Model:
    inp = layers.Input(shape=input_shape)
    x = inp
    x = layers.Conv1D(64, 8, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = residual_block(x, 64, 8)
    x = residual_block(x, 128, 5)
    x = residual_block(x, 128, 3)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train(csv_path: str,
          model_dir: str = 'model_artifacts',
          test_size: float = 0.2,
          random_state: int = 42,
          epochs: int = 100,
          batch_size: int = 32,
          allowed_symbols: list = None):
    os.makedirs(model_dir, exist_ok=True)

    print(f"Lade Daten aus {csv_path}...")
    if allowed_symbols:
        print(f"Filtere auf Symbole: {allowed_symbols}")

    X, y, le = load_data(csv_path, allowed_symbols=allowed_symbols)
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"Daten geladen: {len(X)} Samples, {len(le.classes_)} Klassen ({le.classes_})")

    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        print("Warnung: Zu wenige Daten für Stratified Split (z.B. nur 1 Beispiel pro Klasse). Deaktiviere Stratify...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train_p, scaler = preprocess(X_train)
    X_val_p, _ = preprocess(X_val, scaler=scaler)

    model = build_model(input_shape=X_train_p.shape[1:], n_classes=len(le.classes_))
    model.summary()

    ckpt = callbacks.ModelCheckpoint(os.path.join(model_dir, 'best_model.h5'), save_best_only=True, monitor='val_accuracy', mode='max')
    es = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    hist = model.fit(X_train_p, y_train,
                     validation_data=(X_val_p, y_val),
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=[ckpt, es])

    # Speichere Artefakte
    model.save(os.path.join(model_dir, 'final_model.h5'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.joblib'))

    # Evaluierung & Report
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    preds_prob = model.predict(X_val_p)
    preds = np.argmax(preds_prob, axis=1)
    acc = accuracy_score(y_val, preds)
    # Fix: labels explizit angeben, damit auch Klassen ohne Support im Report auftauchen
    # und kein Mismatch-Fehler entsteht
    all_labels = range(len(le.classes_))
    clf_report = classification_report(y_val, preds, labels=all_labels, target_names=[str(c) for c in le.classes_])
    conf_mat = confusion_matrix(y_val, preds)

    report_txt = os.path.join(model_dir, 'evaluation_report.txt')
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write(f"Validation accuracy: {acc:.6f}\n\n")
        f.write("Classification Report:\n")
        f.write(clf_report)
        f.write("\nConfusion Matrix:\n")
        np.savetxt(f, conf_mat, fmt='%d')

    # Speichere Confusion-Matrix als numpy
    np.save(os.path.join(model_dir, 'confusion_matrix.npy'), conf_mat)

    # Kurze Evaluierung (Keras-eval für Vergleich)
    val_loss, val_acc = model.evaluate(X_val_p, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Evaluation report written to: {report_txt}")
    print("--------------------------------------------------")
    print("TRAINING ABGESCHLOSSEN")
    print("--------------------------------------------------")


def predict_series(series: np.ndarray, model_dir: str = 'model_artifacts') -> Tuple[str, float]:
    """Nimmt ein 1D-Array (gleiche Länge wie Trainings-Timesteps) und gibt (label, prob) zurück."""
    scaler: StandardScaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    le: LabelEncoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
    model = tf.keras.models.load_model(os.path.join(model_dir, 'best_model.h5'))

    series = np.asarray(series, dtype=float).reshape(1, -1)
    series = np.nan_to_num(series, nan=0.0)

    # Überprüfe Länge gegen den Scaler (scaler.n_features_in_ vorhanden in scikit-learn)
    expected = getattr(scaler, 'n_features_in_', None)
    if expected is not None and series.shape[1] != expected:
        raise ValueError(f"Input series length ({series.shape[1]}) stimmt nicht mit erwarteter Länge ({expected}) überein.\n"
                         f"Stelle sicher, dass die Serie genau die gleiche Anzahl Timesteps wie die Trainingsdaten hat.")

    series_p = scaler.transform(series)[..., np.newaxis]

    probs = model.predict(series_p)[0]
    idx = int(np.argmax(probs))
    label = le.inverse_transform([idx])[0]
    return label, float(probs[idx])


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Train or predict time series classifier')
    sub = p.add_subparsers(dest='cmd')

    t = sub.add_parser('train')
    t.add_argument('--csv', default='messdaten.csv')
    t.add_argument('--model-dir', default='model_artifacts')
    t.add_argument('--epochs', type=int, default=50)
    t.add_argument('--symbols', nargs='+', help='Liste der Symbole, auf die trainiert werden soll (z.B. A B C)')
    t.add_argument('--name', help='Name des Modells (speichert in models/NAME)')

    pr = sub.add_parser('predict')
    pr.add_argument('--input', required=True, help='npy file with 1D array or CSV single row')
    # pr.add_argument('--model-dir', default='model_artifacts') # Veraltet

    # Standard-Verhalten: Wenn keine Argumente (Start über Play-Button), dann Training starten
    if len(sys.argv) == 1:
        sys.argv.append('train')

    args = p.parse_args()

    if args.cmd == 'train':
        # Name abfragen wenn nicht gegeben
        model_name = args.name
        if not model_name:
            try:
                raw = input("Bitte Modellnamen eingeben (Enter für 'new_model'): ").strip()
                model_name = raw if raw else "new_model"
            except KeyboardInterrupt:
                sys.exit(0)
        
        # Pfad zusammenbauen
        save_path = os.path.join("models", model_name)
        print(f"Modell wird gespeichert unter: {save_path}")
        
        train(args.csv, model_dir=save_path, epochs=args.epochs, allowed_symbols=args.symbols)
        
    elif args.cmd == 'predict':
        inp = args.input
        if inp.lower().endswith('.npy'):
            s = np.load(inp)
        else:
            # versuche CSV einzeilige Reihe zu laden (kommav/semikolon-handling)
            df = pd.read_csv(inp, sep=None, engine='python')
            s = df.values.flatten()
        label, prob = predict_series(s, model_dir=args.model_dir)
        print(f"Predicted: {label} (prob={prob:.4f})")
    else:
        p.print_help()
