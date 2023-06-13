import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données à partir du fichier CSV
data = pd.read_csv('./data/data.csv')

# Diviser les données en fonction des caractéristiques (X) et de la variable cible (y)
X = data.iloc[:, :-1]  # Sélectionner toutes les colonnes sauf la dernière
y = data.iloc[:, -1]   # Sélectionner la dernière colonne

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle Random Forest Classifier
model = RandomForestClassifier()

# Entraîner le modèle avec les données d'entraînement
model.fit(X_train, y_train)

# Faire des prédictions sur les données de test
y_pred = model.predict(X_test)

# Calculer l'exactitude
accuracy = accuracy_score(y_test, y_pred)
print("Exactitude :", accuracy)