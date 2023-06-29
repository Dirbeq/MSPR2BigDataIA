from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def regression_lineaire(x_data_names=None, y_data_names=None):
    print("----------------------- Régression linéaire -----------------------")
    # Diviser les données en caractéristiques (X) et étiquettes (y)
    X = x_data_names
    y = y_data_names

    # Diviser les données en jeux d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer le modèle de régression linéaire avec descente de gradient
    model = SGDRegressor(max_iter=100)

    # Entraîner le modèle sur les données d'entraînement
    print("Entraînement du modèle...")
    model.fit(x_train, y_train)
    print("Entraînement terminé.")

    # Faire des prédictions sur les données de test
    print("Prédiction sur les données de test...")
    y_pred = model.predict(x_test)
    print("Prédiction terminée.")

    # Calculer l'erreur quadratique moyenne (RMSE)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE du modèle : {rmse}")

    # Accuracy
    accuracy = model.score(x_test, y_test)
    print(f"Précision du modèle : {accuracy}")

    print("----------------------- Fin régression linéaire -----------------------")
