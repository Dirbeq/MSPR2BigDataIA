from sklearn.model_selection import train_test_split

from prediction_model.PyUtils import calculate_scores


def perform_on_model(model, x_data, y_data, test_size=0.2):
    X = x_data
    y = y_data

    model_name = type(model).__name__

    print(f"----------------------- {model_name} -----------------------")
    print(f"Ratio de test : {test_size}")

    # Diviser les données en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Entraîner le modèle sur l'ensemble d'entraînement
    print("Entraînement du modèle...")
    model.fit(x_train, y_train)

    # Prédire les étiquettes sur l'ensemble d'entraînement
    y_pred_train = model.predict(x_train)

    # Calculer les scores sur l'ensemble d'entraînement
    print("Calcul des scores sur l'ensemble d'entraînement...")
    calculate_scores(y_train, y_pred_train, y.value_counts().index.tolist(), model_name=model_name)

    # Prédire les étiquettes sur l'ensemble de test
    y_pred_test = model.predict(x_test)

    # Calculer les scores sur l'ensemble de test
    print("Calcul des scores sur l'ensemble de test...")
    calculate_scores(y_test, y_pred_test, y.value_counts().index.tolist(), model_name=model_name)

    print(f"----------------------- Fin {model_name} -----------------------")
