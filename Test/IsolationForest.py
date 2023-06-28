import pandas as pd
from sklearn.ensemble import IsolationForest


def isolation_forest(x_data_names=None, y_data_names=None):
    print("----------------------- Isolation Forest -----------------------")
    # Diviser les données en caractéristiques (X) et étiquettes (y)
    X = x_data_names
    y = y_data_names

    # Créer le modèle Isolation Forest
    model = IsolationForest(n_estimators=100, random_state=42)

    # Entraîner le modèle
    print("Entraînement du modèle...")
    model.fit(X)

    # Prédire les anomalies sur les données d'entraînement
    y_pred_train = model.predict(X)

    # Accuracy for Isolation Forest
    accuracy = list(y_pred_train).count(1) / y_pred_train.shape[0]
    print(f"Précision du modèle : {accuracy}")

    # Prédire les anomalies sur de nouvelles données
    new_data = pd.DataFrame([[10990,13.1,52.9,9.9], [24030,5.5,10.5,3.4]],
                            columns=['Médiane du niveau de vie 2020', 'Taux de chômage annuel moyen 2022',
                                     'Taux de pauvreté 2020',
                                     'Coups et blessures volontaires (taux) 2022'])
    print("Prédiction sur de nouvelles valeurs...")
    predictions = model.predict(new_data)
    # Accuracy for Isolation Forest
    accuracy = list(predictions).count(1) / predictions.shape[0]
    print(f"Précision du modèle : {accuracy}")

    print("----------------------- Fin de l'Isolation Forest -----------------------")
