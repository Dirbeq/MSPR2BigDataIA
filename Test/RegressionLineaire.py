import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# Charger les données à partir du fichier CSV
data = pd.read_csv('../data/Data1.csv',on_bad_lines='skip', sep=";", index_col=0)

# Diviser les données en caractéristiques (X) et étiquettes (y)
X = data[['Code de la circonscription', '% Abs/Ins']]
y = data['N°Panneau']

# Diviser les données en jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle de régression linéaire avec descente de gradient
model = SGDRegressor(max_iter=100)

# Entraîner le modèle sur les données d'entraînement
print("Entraînement du modèle...")
model.fit(X_train, y_train)
print("Entraînement terminé.")

# Faire des prédictions sur les données de test
print("Prédiction sur les données de test...")
y_pred = model.predict(X_test)
print("Prédiction terminée.")

# Calculer l'erreur quadratique moyenne (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE du modèle : {rmse}")

# Prédire de nouvelles valeurs
new_data = pd.DataFrame([[26080, 12.2], [75078, 23.8]], columns=['Code de la circonscription', '% Abs/Ins'])
print("Prédiction sur de nouvelles valeurs...")
predictions = model.predict(new_data)
print("Prédiction terminée.")
print(f"Prédictions : {predictions}")
