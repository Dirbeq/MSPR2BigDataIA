import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def svm(x_data_names=None, y_data_names=None):
    print("----------------------- SVM -----------------------")
    # Diviser les données en caractéristiques (X) et étiquettes (y)
    X = x_data_names
    y = y_data_names

    # Diviser les données en jeux d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Spécifier le paramètre verbose
    verbose = True

    # Créer le modèle SVM avec le paramètre verbose spécifié
    print("Création du modèle SVM")
    model = SVC(verbose=verbose)

    # Entraîner le modèle sur les données d'entraînement
    print("Entraînement du modèle SVM")
    model.fit(x_train, y_train)

    # Faire des prédictions sur les données de test
    print("Prédictions sur les données de test")
    y_pred = model.predict(x_test)

    # Calculer la précision du modèle
    print("Calcul de la précision du modèle")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision du modèle : {accuracy}")

    # Prédire de nouvelles valeurs
    new_data = pd.DataFrame([[10990, 13.1, 52.9, 9.9], [24030, 5.5, 10.5, 3.4]],
                            columns=['Médiane du niveau de vie 2020', 'Taux de chômage annuel moyen 2022',
                                     'Taux de pauvreté 2020',
                                     'Coups et blessures volontaires (taux) 2022'])
    predictions = model.predict(new_data)
    print(f"Prédictions : {predictions}")
    # Calculer l'accuracy sur les nouvelles données
    accuracy = accuracy_score([0, 2], predictions)  # Remplacez [0, 1] par les véritables étiquettes si connues
    print(f"Précision du modèle sur de nouvelles valeurs : {accuracy}")

    print("----------------------- Fin SVM -----------------------")
