import pandas as pd
from sklearn.ensemble import IsolationForest

# Charger les données à partir du fichier CSV
datad = pd.read_csv('../data/Data1.csv',on_bad_lines='skip', sep=";", index_col=0)


def isolation_forest(data=datad):
    # Diviser les données en caractéristiques (X) et étiquettes (y)
    X = data[['Code de la circonscription', '% Abs/Ins']]
    y = data['N°Panneau']

    # Créer le modèle Isolation Forest
    model = IsolationForest(n_estimators=100, random_state=42)

    # Entraîner le modèle
    print("Entraînement du modèle...")
    model.fit(X)

    # Prédire les anomalies sur les données d'entraînement
    y_pred_train = model.predict(X)

    # Accuracy
    accuracy = model.score(X, y)
    print(f"Précision du modèle : {accuracy}")

    # Prédire les anomalies sur de nouvelles données
    new_data = pd.DataFrame([[1, 75.2], [2, 65.8]], columns=['Code de la circonscription', '% Abs/Ins'])
    print("Prédiction sur de nouvelles valeurs...")
    y_pred_new = model.predict(new_data)

    print("Prédictions sur les données d'entraînement:")
    print(y_pred_train)
    print("Prédictions sur les nouvelles données:")
    print(y_pred_new)
