import pandas as pd

import matplotlib.pyplot as plt

# Charger le fichier CSV
df = pd.read_csv('progress.csv')

# Vérifier si la colonne existe (parfois il peut y avoir des espaces)
col = [c for c in df.columns if 'ep_rew_mean' in c][0]

# Tracer la courbe
plt.figure(figsize=(10, 5))
plt.plot(df[col])
plt.xlabel('Itérations')
plt.ylabel('Récompense Moyenne par Episode')
plt.title('Évolution de la récompense moyenne par épisode')
plt.grid(True)
plt.tight_layout()
plt.show()