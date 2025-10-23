#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_experiments_v11a.py - Test avec seuil à 0.3 (plus agressif)

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Ajouter les chemins pour les imports locaux
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(script_dir, "stocknet-code")
sys.path.insert(0, script_dir)
sys.path.insert(0, code_dir)

# Imports conditionnels des modules locaux
try:
    from vol_price_ratio import calculate_vol_price_ratio
    from ctx_div_cvd_uptrend import calculate_ctx_div_cvd_uptrend
    from ctx_div_rsi_downtrend import calculate_ctx_div_rsi_downtrend
    from ctx_div_cvd_lowvol import calculate_ctx_div_cvd_lowvol
    from ctx_div_rsi_lowvol import calculate_ctx_div_rsi_lowvol
    from cvd_state import calculate_cvd_state
    MODULES_OK = True
except ImportError as e:
    print(f"⚠️ Modules locaux non trouvés: {e}")
    MODULES_OK = False

# --- CONFIG CPU ONLY ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.get_logger().setLevel("ERROR")

print("\n=== StockNet v11a – Seuil agressif 0.3 ===")

if not MODULES_OK:
    print("\n❌ Impossible de continuer sans les modules requis.")
    sys.exit(1)

# --- CHARGEMENT DES DONNÉES ---
df = pd.read_parquet("/home/jx/Documents/stocknet/out/features/BTC_features_v9.parquet")
df = calculate_vol_price_ratio(df)
df = calculate_ctx_div_cvd_uptrend(df)
df = calculate_ctx_div_rsi_downtrend(df)
df = calculate_ctx_div_cvd_lowvol(df)
df = calculate_ctx_div_rsi_lowvol(df)
print(f"✅ Données : {len(df):,} lignes")

# === INTÉGRATION AUTOMATIQUE DU CVD_STATE ===
df = calculate_cvd_state(df)
print("✅ Colonne cvd_state générée avant feature shifting.")

# --- AJOUT DU TARGET AVEC SEUIL 0.3 (PLUS AGRESSIF) ---
print("\n🎯 Génération du target avec seuil 0.3 × volatilité...")

# Calcul des returns futurs
returns = df["close"].pct_change().shift(-1)

# Seuil dynamique basé sur la volatilité locale (fenêtre de 100 périodes)
rolling_vol = returns.rolling(window=100, min_periods=50).std()
threshold = rolling_vol * 0.3  # ⚡ CHANGEMENT ICI : 0.3 au lieu de 0.5

# Classification 3-classes
df["target"] = np.where(
    returns > threshold, 2,           # Classe 2: Strong Buy
    np.where(returns < -threshold, 0, 1)  # Classe 0: Strong Sell / Classe 1: Hold
)

df = df.dropna()

# --- DIAGNOSTIC DE DISTRIBUTION ---
print("\n📊 Distribution des classes (avant split) :")
print(df["target"].value_counts().sort_index())
print("\nProportions :")
print(df["target"].value_counts(normalize=True).sort_index())

# --- LISTE DES FEATURES ---
features = [
    "ret","rsi_14","ema_20","ema_50","volatility_zscore","trend_state",
    "vol_state","ctx_confluence","hour_sin","hour_cos",
    "delta_price","delta_vol","ofi_proxy","cvd_proxy","delta_power",
    "spread","price_change","vol_imbalance","volatility_20","cvd_state"
]

# --- SHIFTING ANTI-LEAK ---
for f in features:
    if f in df.columns:
        df[f] = df[f].shift(1)
    else:
        print(f"⚠️ Feature manquante ignorée: {f}")
df = df.dropna()

# --- NORMALISATION ---
if "cvd_state" in df.columns:
    mapping = {"Weak": -1, "Neutral": 0, "Strong": 1}
    df["cvd_state"] = df["cvd_state"].map(mapping)
    print("✅ cvd_state encodé numériquement.")

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# --- SPLIT TRAIN / TEST ---
split_idx = int(len(df) * 0.75)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]
gap = int(len(df) * 0.05)
test = test.iloc[gap:]

print(f"\n⏱️ Split strict: train={len(train)}, test={len(test)}, gap={gap}")

# --- FENÊTRAGE ---
def make_windows(data, horizon=100):
    X, y = [], []
    vals, labels = data[features].values, data["target"].values
    for i in range(0, len(vals)-horizon-1, horizon):
        X.append(vals[i:i+horizon])
        y.append(labels[i+horizon])
    return np.array(X), np.array(y)

X_train, y_train = make_windows(train)
X_test, y_test = make_windows(test)

print(f"✅ Fenêtrage: X_train={X_train.shape}, X_test={X_test.shape}")

# --- DIAGNOSTIC POST-WINDOWING ---
from collections import Counter
print("\n📊 Distribution après windowing :")
print("Train:", Counter(y_train))
print("Test:", Counter(y_test))

# --- CALCUL DES POIDS DE CLASSE ---
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\n⚖️ Poids de classe calculés: {class_weight_dict}")

# --- DÉFINITION DU MODÈLE (IDENTIQUE À V10) ---
def StockNetCNN_WaveNet(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(64, 3, padding="causal", dilation_rate=2, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(3, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

model = StockNetCNN_WaveNet((100, len(features)))
print("\n⚙️ Entraînement StockNetCNN+WaveNet (3-classes)...")

cb = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=1,
    callbacks=cb,
    class_weight=class_weight_dict
)

# --- ÉVALUATION ---
loss, acc = model.evaluate(X_test, y_test)
print(f"\n📊 Accuracy={acc:.3f}")

if acc < 0.45:
    print("⚠️ Accuracy < 45% → Besoin du modèle plus complexe (Option C)")
else:
    print("✅ Accuracy ≥ 45% → Le seuil 0.3 fonctionne bien !")

# --- SAUVEGARDE ---
model.save("/home/jx/Documents/stocknet/out/model_v11a.keras")
print("�� Modèle sauvegardé : model_v11a.keras")

# --- SAUVEGARDE DES PRÉDICTIONS ---
from sklearn.metrics import classification_report, confusion_matrix

print("\n📈 Génération des prédictions et rapport de test...")
y_pred = np.argmax(model.predict(X_test), axis=1)

# Enregistrement CSV
results = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
out_path = "/home/jx/Documents/stocknet/out/preds_v11a.csv"
results.to_csv(out_path, index=False)
print(f"💾 Résultats enregistrés : {out_path}")

# Rapport complet
print("\n📊 Rapport de classification :")
print(classification_report(y_test, y_pred, digits=3, target_names=["Sell", "Hold", "Buy"]))
print("\n🧩 Matrice de confusion :")
print("        Sell  Hold  Buy")
cm = confusion_matrix(y_test, y_pred)
for i, row in enumerate(cm):
    print(f"{['Sell', 'Hold', 'Buy'][i]:5s} {row}")
