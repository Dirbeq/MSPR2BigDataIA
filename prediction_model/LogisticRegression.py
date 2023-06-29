from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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

    # Prédire les étiquettes sur l'ensemble d'entraînement
    y_pred_train = model.predict(x_train)

    # Calculer l'accuracy sur l'ensemble d'entraînement
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print(f"Précision du modèle sur l'ensemble d'entraînement : {accuracy_train}")

    # Prédire les étiquettes sur l'ensemble de test
    y_pred_test = model.predict(x_test)

    # Calculer l'accuracy sur l'ensemble de test
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Précision du modèle sur l'ensemble de test : {accuracy_test}")


print("----------------------- Fin régression logistique -----------------------")
