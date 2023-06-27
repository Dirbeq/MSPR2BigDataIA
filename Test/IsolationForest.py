import pandas as pd
from sklearn.ensemble import IsolationForest


def isolation_forest(data=None, x_data_names=None, y_data_names=None):
    # Diviser les données en caractéristiques (X) et étiquettes (y)
    X = data[x_data_names]
    y = data[y_data_names]

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
