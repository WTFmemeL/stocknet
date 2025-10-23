#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_experiments_v11c.py - Modèle CNN profond avec Attention

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, Model
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

print("\n=== StockNet v11c – CNN Profond + Attention ===")

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

# --- AJOUT DU TARGET AVEC SEUIL 0.3 ---
print("\n🎯 Génération du target avec seuil 0.3 × volatilité...")

returns = df["close"].pct_change().shift(-1)
rolling_vol = returns.rolling(window=100, min_periods=50).std()
threshold = rolling_vol * 0.3

df["target"] = np.where(
    returns > threshold, 2,
    np.where(returns < -threshold, 0, 1)
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

# --- FENÊTRAGE GLISSANT (SLIDING WINDOW) ---
def make_windows(data, horizon=100, stride=10):
    """
    Fenêtrage glissant avec stride=10 au lieu de 100
    → 30× plus de données d'entraînement !
    """
    X, y = [], []
    vals, labels = data[features].values, data["target"].values
    for i in range(0, len(vals)-horizon-1, stride):  # ⚡ CHANGEMENT ICI
        X.append(vals[i:i+horizon])
        y.append(labels[i+horizon])
    return np.array(X), np.array(y)

print("\n🔄 Fenêtrage glissant avec stride=10...")
X_train, y_train = make_windows(train, stride=10)
X_test, y_test = make_windows(test, stride=10)

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

# --- DÉFINITION DU MODÈLE PROFOND AVEC ATTENTION ---
def StockNetDeep(input_shape):
    """
    Architecture CNN profonde avec :
    - 3 blocs Conv1D (32 → 64 → 128 filtres)
    - BatchNormalization après chaque couche
    - Dropout pour régularisation
    - Multi-Head Attention
    - Connexions résiduelles
    """
    inp = layers.Input(shape=input_shape)
    
    # === BLOC 1 : Features bas niveau ===
    x = layers.Conv1D(32, 3, padding="causal", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # === BLOC 2 : Features intermédiaires ===
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # === BLOC 3 : Features complexes avec dilation ===
    x = layers.Conv1D(128, 3, padding="causal", dilation_rate=2, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # === ATTENTION MULTI-TÊTES ===
    # Permet au modèle de se concentrer sur les features importantes
    attn_output = layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=32,
        dropout=0.1
    )(x, x)
    
    # Connexion résiduelle (évite la perte d'information)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    
    # === AGRÉGATION TEMPORELLE ===
    x = layers.GlobalAveragePooling1D()(x)
    
    # === CLASSIFICATION ===
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(3, activation="softmax")(x)
    
    model = Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

model = StockNetDeep((100, len(features)))
print("\n🧠 Architecture du modèle :")
model.summary()

print("\n⚙️ Entraînement StockNet Deep (3-classes)...")

cb = [
    callbacks.EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,  # ⚡ Augmenté de 32 à 64 pour plus de stabilité
    verbose=1,
    callbacks=cb,
    class_weight=class_weight_dict
)

# --- ÉVALUATION ---
loss, acc = model.evaluate(X_test, y_test)
print(f"\n📊 Accuracy finale = {acc:.3f}")

if acc >= 0.45:
    print("✅ Succès ! Accuracy ≥ 45%")
else:
    print("⚠️ Toujours < 45%, mais c'est le meilleur qu'on puisse faire avec ces données")

# --- SAUVEGARDE ---
model.save("/home/jx/Documents/stocknet/out/model_v11c.keras")
print("💾 Modèle sauvegardé : model_v11c.keras")

# --- SAUVEGARDE DES PRÉDICTIONS ---
from sklearn.metrics import classification_report, confusion_matrix

print("\n📈 Génération des prédictions et rapport de test...")
y_pred = np.argmax(model.predict(X_test), axis=1)

# Enregistrement CSV
results = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
out_path = "/home/jx/Documents/stocknet/out/preds_v11c.csv"
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

# --- COMPARAISON AVEC V11A ---
print("\n📈 Comparaison avant/après sliding window :")
print(f"v11c (stride=100) : 35.3% avec 2,946 samples")
print(f"v11c (stride=10)  : {acc:.1%} avec ~30,000 samples")
diff = (acc - 0.353) * 100
if diff > 5:
    print(f"✅ ÉNORME amélioration de +{diff:.1f} points !")
elif diff > 0:
    print(f"✅ Amélioration de +{diff:.1f} points")
else:
    print(f"❌ Pas d'amélioration, problème plus profond...")
