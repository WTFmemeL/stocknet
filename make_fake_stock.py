import os
import pandas as pd
import numpy as np

# Chemin cible
folder = "stocknet-dataset/data/asset_raw_splited_series"
os.makedirs(folder, exist_ok=True)

# Données fictives
n = 2000
dates = pd.date_range("2020-01-01", periods=n, freq="D")
data = {
    "close": np.cumsum(np.random.randn(n)) + 100,
    "pos": np.random.rand(n),
    "neg": np.random.rand(n),
    "lit": np.random.rand(n),
    "unc": np.random.rand(n),
    "con": np.random.rand(n),
    "see": np.random.rand(n)
}
df = pd.DataFrame(data, index=dates)

# Sauvegarde
df.to_pickle(f"{folder}/AAPL.pkl")
print(f"✅ Fichier synthétique créé : {folder}/AAPL.pkl ({len(df)} lignes)")
