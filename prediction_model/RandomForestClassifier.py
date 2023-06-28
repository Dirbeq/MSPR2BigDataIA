import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def random_forest_classifier(x_data_names=None, y_data_names=None):
    print("----------------------- Random Forest Classifier -----------------------")
    # Diviser les données en caractéristiques (X) et étiquettes (y)
    X = x_data_names
    y = y_data_names

    # Créer le modèle Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Entraîner le modèle
    print("Entraînement du modèle...")
    model.fit(X, y)

    # Prédire les étiquettes sur les données d'entraînement
    y_pred_train = model.predict(X)

    # Calculer l'accuracy sur les données d'entraînement
    accuracy = accuracy_score(y, y_pred_train)
    print(f"Précision du modèle sur les données d'entraînement : {accuracy}")

    # Prédire les étiquettes sur de nouvelles données
    new_data = pd.DataFrame([[10990, 13.1, 52.9, 9.9], [24030, 5.5, 10.5, 3.4]],
                            columns=['Médiane du niveau de vie 2020', 'Taux de chômage annuel moyen 2022',
                                     'Taux de pauvreté 2020',
                                     'Coups et blessures volontaires (taux) 2022'])
    print("Prédiction sur de nouvelles valeurs...")
    predictions = model.predict(new_data)

    # Calculer l'accuracy sur les nouvelles données
    accuracy = accuracy_score([0, 1], predictions)  # Remplacez [0, 1] par les véritables étiquettes si connues
    print(f"Précision du modèle sur de nouvelles valeurs : {accuracy}")

    print("----------------------- Fin du Random Forest Classifier -----------------------")
