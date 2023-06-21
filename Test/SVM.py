import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Charger les données à partir du fichier CSV
data = pd.read_csv('../data/Data1.csv',on_bad_lines='skip', sep=";", index_col=0)

# Diviser les données en caractéristiques (X) et étiquettes (y)
X = data[['Code de la circonscription', '% Abs/Ins']]
y = data['N°Panneau']

# Diviser les données en jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Spécifier le paramètre verbose
verbose = True

# Créer le modèle SVM avec le paramètre verbose spécifié
print("Création du modèle SVM")
model = SVC(verbose=verbose)

# Entraîner le modèle sur les données d'entraînement
print("Entraînement du modèle SVM")
model.fit(X_train, y_train)

# Faire des prédictions sur les données de test
print("Prédictions sur les données de test")
y_pred = model.predict(X_test)

# Calculer la précision du modèle
print("Calcul de la précision du modèle")
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy}")

# Prédire de nouvelles valeurs
new_data = pd.DataFrame([[26080, 12.2], [75078, 23.8]], columns=['Code de la circonscription', '% Abs/Ins'])
predictions = model.predict(new_data)
print(f"Prédictions : {predictions}")
