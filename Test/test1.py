import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def logistic_regression(data=None, x_data_names=None, y_data_names=None):
    print("----------------------- Régression logistique -----------------------")
    # Diviser les données en caractéristiques (X) et étiquettes (y)
    X = x_data_names
    y = y_data_names

    # Diviser les données en jeux d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer le modèle
    model = LogisticRegression()

    # Spécifier le nombre d'époques
    num_epochs = 10
    print(f"Nombre d'époques : {num_epochs}")

    # Entraîner le modèle sur les données d'entraînement avec le nombre d'époques spécifié
    for epoch in tqdm(range(num_epochs)):
        model.fit(x_train, y_train)

    # Évaluer la précision du modèle sur les données de test
    accuracy = model.score(x_test, y_test)
    print(f"Précision du modèle : {accuracy}")

    # Prédire de nouvelles valeurs
    new_data = pd.DataFrame([[10990, 13.1, 52.9, 9.9], [24030, 5.5, 10.5, 3.4]],
                            columns=['Médiane du niveau de vie 2020', 'Taux de chômage annuel moyen 2022',
                                     'Taux de pauvreté 2020',
                                     'Coups et blessures volontaires (taux) 2022'])
    predictions = model.predict(new_data)
    print(f"Prédictions : {predictions}")

    print("----------------------- Fin régression logistique -----------------------")
