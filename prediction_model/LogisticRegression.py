from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def logistic_regression(x_data_names=None, y_data_names=None):
    print("----------------------- Régression logistique -----------------------")
    # Diviser les données en caractéristiques (X) et étiquettes (y)
    X = x_data_names
    y = y_data_names

    # Diviser les données en jeux d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer le modèle
    model = LogisticRegression(max_iter=300)

    model.fit(x_train, y_train)

    # Évaluer la précision du modèle sur les données de test
    accuracy = model.score(x_test, y_test)
    print(f"Précision du modèle : {accuracy}")

    print("----------------------- Fin régression logistique -----------------------")
