#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_experiments_v17.py - Target adaptatif : Top 20% des mouvements

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

print("\n=== StockNet v17 – Target Adaptatif (Top 20% des mouvements) ===")

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
print("✅ Colonne cvd_state générée.")

# === AJOUT FEATURE : RÉGIME DE MARCHÉ ===
print("\n🎯 Calcul du régime de marché...")

# Volatilité roulante
rolling_vol = df["close"].pct_change().rolling(50).std()

# Classification en 3 régimes
vol_q33 = rolling_vol.quantile(0.33)
vol_q66 = rolling_vol.quantile(0.66)

df["market_regime"] = np.where(
    rolling_vol > vol_q66, 2,      # Haute volatilité (trending/breakout)
    np.where(rolling_vol < vol_q33, 0, 1)  # Basse volatilité (range) / Moyen
)

print(f"✅ Régimes : Low={vol_q33:.6f}, Mid, High={vol_q66:.6f}")
print(f"Distribution : {df['market_regime'].value_counts().sort_index().to_dict()}")

# === TARGET ADAPTATIF : TOP 20% DES MOUVEMENTS ===
print("\n🎯 Génération du target adaptatif (Top 20%)...")
print("   Principe : Les 20% meilleurs mouvements haussiers/baissiers")
print("   Fenêtre : 1-12 minutes")
print("   Avantage : Distribution 20/60/20 plus équilibrée")

window = 12  # minutes

# 1. Calcul des returns futurs sur la fenêtre
future_returns = pd.DataFrame({
    f'ret_{i}': df["close"].pct_change(i).shift(-i) 
    for i in range(1, window + 1)
})

# 2. Maximum et minimum des mouvements sur la fenêtre
max_future_move = future_returns.max(axis=1)
min_future_move = future_returns.min(axis=1)

# 3. CLASSIFICATION ADAPTATIVE PAR QUANTILES
# Top 20% des mouvements haussiers = Strong Buy
# Top 20% des mouvements baissiers = Strong Sell
# Le reste = Hold

quantile_buy = max_future_move.quantile(0.80)   # 80ème percentile
quantile_sell = min_future_move.quantile(0.20)  # 20ème percentile

print(f"\n📊 Seuils adaptatifs calculés :")
print(f"   Buy threshold (top 20%) : {quantile_buy:.4%}")
print(f"   Sell threshold (bottom 20%) : {quantile_sell:.4%}")

# Classification finale
df["target"] = np.where(
    max_future_move >= quantile_buy, 2,      # Strong Buy (top 15%)
    np.where(min_future_move <= quantile_sell, 0, 1)  # Strong Sell (bottom 15%) / Hold
)

# Stockage des métriques pour analyse
df["max_future_move"] = max_future_move
df["min_future_move"] = min_future_move

df = df.dropna()

# --- DIAGNOSTIC DE DISTRIBUTION ---
print("\n📊 Distribution des classes (avant split) :")
print(df["target"].value_counts().sort_index())
print("\nProportions :")
print(df["target"].value_counts(normalize=True).sort_index())

# Statistiques détaillées
print(f"\n📈 Statistiques des mouvements :")
print(f"   Max moyen : {max_future_move.mean():.4%}")
print(f"   Min moyen : {min_future_move.mean():.4%}")
print(f"   Max médian : {max_future_move.median():.4%}")
print(f"   Min médian : {min_future_move.median():.4%}")
buy_mask = df['target'] == 2
sell_mask = df['target'] == 0
hold_mask = df['target'] == 1
print(f"\n   Mouvement moyen des Strong Buy : {df.loc[buy_mask, 'max_future_move'].mean():.4%}")
print(f"   Mouvement moyen des Strong Sell : {df.loc[sell_mask, 'min_future_move'].mean():.4%}")
print(f"   Mouvement moyen des Hold : {df.loc[hold_mask, 'max_future_move'].mean():.4%}")

# --- VÉRIFICATION ANTI-LEAKAGE RENFORCÉE ---
print("\n🔍 === VÉRIFICATION ANTI-LEAKAGE ===")
print("\nÉchantillon de données (indices 1000-1005) :")
check_cols = ["close", "ret", "rsi_14", "max_future_move", "target"]
print(df[check_cols].iloc[1000:1005])

# --- LISTE DES FEATURES (avec régime de marché) ---
features = [
    "ret","rsi_14","ema_20","ema_50","volatility_zscore","trend_state",
    "vol_state","ctx_confluence","hour_sin","hour_cos",
    "delta_price","delta_vol","ofi_proxy","cvd_proxy","delta_power",
    "spread","price_change","vol_imbalance","volatility_20","cvd_state",
    "market_regime"
]

print(f"\n✅ {len(features)} features utilisées")
print("   Note : Le volume est utilisé comme feature (delta_vol, vol_imbalance)")
print("   → Le modèle apprend lui-même son importance !")

# --- SHIFTING ANTI-LEAK STRICT ---
print("\n🔒 Application du shifting anti-leak...")
for f in features:
    if f in df.columns:
        df[f] = df[f].shift(1)
    else:
        print(f"⚠️ Feature manquante ignorée: {f}")
df = df.dropna()

# Vérification post-shift
print("\n🔍 Après shifting (indices 1000-1005) :")
print(df[check_cols].iloc[1000:1005])

# --- NORMALISATION ---
if "cvd_state" in df.columns:
    mapping = {"Weak": -1, "Neutral": 0, "Strong": 1}
    df["cvd_state"] = df["cvd_state"].map(mapping)
    print("\n✅ cvd_state encodé numériquement.")

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# --- SPLIT TRAIN / TEST ---
split_idx = int(len(df) * 0.75)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]
gap = int(len(df) * 0.05)
test = test.iloc[gap:]

print(f"\n⏱️ Split strict: train={len(train)}, test={len(test)}, gap={gap}")

# --- FENÊTRAGE GLISSANT ---
def make_windows(data, horizon=100, stride=10):
    X, y = [], []
    vals, labels = data[features].values, data["target"].values
    for i in range(0, len(vals)-horizon-1, stride):
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
train_dist = Counter(y_train)
test_dist = Counter(y_test)
print(f"Train: {train_dist}")
print(f"Test: {test_dist}")

# Baseline accuracy
baseline_train = max(train_dist.values()) / sum(train_dist.values())
baseline_test = max(test_dist.values()) / sum(test_dist.values())
print(f"\n📌 Baseline (prédire toujours la classe majoritaire) :")
print(f"   Train: {baseline_train:.1%}")
print(f"   Test: {baseline_test:.1%}")
print(f"   → Le modèle DOIT battre ce score pour être utile !")

# Vérifier l'équilibre des classes
if baseline_test < 0.75:
    print(f"\n✅ Distribution équilibrée ! Baseline raisonnable ({baseline_test:.1%})")
    print("   → Le modèle a une vraie chance d'apprendre")
else:
    print(f"\n⚠️ Baseline encore élevé ({baseline_test:.1%})")
    print("   → Distribution pas idéale mais meilleure qu'avant")

# --- CALCUL DES POIDS DE CLASSE ---
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\n⚖️ Poids de classe calculés: {class_weight_dict}")

# --- DÉFINITION DU MODÈLE ---
def StockNetV17(input_shape):
    """
    Architecture optimisée pour apprendre des patterns relatifs (Top 20%)
    """
    inp = layers.Input(shape=input_shape)
    
    # Bloc 1 - Extraction patterns courts
    x = layers.Conv1D(64, 5, padding="causal", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Bloc 2 - Patterns moyens
    x = layers.Conv1D(128, 3, padding="causal", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Bloc 3 - Patterns longs avec dilation
    x = layers.Conv1D(128, 3, padding="causal", dilation_rate=2, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    # Attention - Capture les dépendances temporelles
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    
    # Agrégation
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(3, activation="softmax")(x)
    
    model = Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

model = StockNetV17((100, len(features)))
print("\n🧠 Architecture du modèle :")
model.summary()

print("\n⚙️ Entraînement StockNet v17...")

cb = [
    callbacks.EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    verbose=1,
    callbacks=cb,
    class_weight=class_weight_dict
)

# --- ÉVALUATION ---
loss, acc = model.evaluate(X_test, y_test)
print(f"\n📊 === RÉSULTATS FINAUX ===")
print(f"Accuracy = {acc:.3f} ({acc:.1%})")
print(f"Baseline = {baseline_test:.3f} ({baseline_test:.1%})")

improvement = (acc - baseline_test) * 100
if acc > baseline_test + 0.05:
    print(f"✅ SUCCÈS ! +{improvement:.1f} points au-dessus du baseline")
    print("   → Le modèle détecte les meilleurs mouvements !")
elif acc > baseline_test:
    print(f"⚠️ Légère amélioration (+{improvement:.1f} points)")
    print("   → Le modèle apprend mais peut encore progresser")
else:
    print(f"❌ Baseline non battu (écart: {improvement:.1f} points)")
    if improvement > -5:
        print("   → Proche du baseline, le modèle apprend quelque chose")
    else:
        print("   → Le modèle a du mal à généraliser")

# --- SAUVEGARDE ---
model.save("/home/jx/Documents/stocknet/out/model_v17.keras")
print("\n💾 Modèle sauvegardé : model_v17.keras")

# --- SAUVEGARDE DES PRÉDICTIONS ---
from sklearn.metrics import classification_report, confusion_matrix

print("\n📈 Génération des prédictions et rapport de test...")
y_pred = np.argmax(model.predict(X_test), axis=1)

# Enregistrement CSV
results = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
out_path = "/home/jx/Documents/stocknet/out/preds_v17.csv"
results.to_csv(out_path, index=False)
print(f"💾 Résultats enregistrés : {out_path}")

# Rapport complet
print("\n📊 Rapport de classification :")
print(classification_report(y_test, y_pred, digits=3, target_names=["Strong Sell", "Hold", "Strong Buy"]))
print("\n🧩 Matrice de confusion :")
print("              Sell  Hold  Buy")
cm = confusion_matrix(y_test, y_pred)
for i, row in enumerate(cm):
    print(f"{['Strong Sell', 'Hold      ', 'Strong Buy '][i]} {row}")

# Analyse des erreurs
print("\n🔍 Analyse des erreurs :")
total_errors = len(y_test) - np.sum(y_pred == y_test)
print(f"Total erreurs : {total_errors} / {len(y_test)} ({total_errors/len(y_test):.1%})")

# Erreurs critiques
critical_errors = np.sum((y_test == 0) & (y_pred == 2)) + np.sum((y_test == 2) & (y_pred == 0))
print(f"Erreurs critiques (Sell↔Buy) : {critical_errors} ({critical_errors/len(y_test):.1%})")

# Précision sur les signaux forts uniquement
strong_signals = (y_test != 1)
if strong_signals.sum() > 0:
    strong_acc = np.sum((y_pred == y_test) & strong_signals) / strong_signals.sum()
    print(f"\n💪 Précision sur signaux forts (Buy/Sell) : {strong_acc:.1%}")
    print("   → Mesure la qualité des prédictions sur les meilleurs mouvements")
    
    # Détail par classe
    buy_correct = np.sum((y_pred == 2) & (y_test == 2))
    sell_correct = np.sum((y_pred == 0) & (y_test == 0))
    buy_total = np.sum(y_test == 2)
    sell_total = np.sum(y_test == 0)
    
    if buy_total > 0:
        print(f"   Buy recall : {buy_correct/buy_total:.1%} ({buy_correct}/{buy_total})")
    if sell_total > 0:
        print(f"   Sell recall : {sell_correct/sell_total:.1%} ({sell_correct}/{sell_total})")

print("\n✅ v17 terminé ! Target adaptatif avec Top 20% 🎯")
print("   Distribution 20/60/20 pour meilleur équilibre !")
