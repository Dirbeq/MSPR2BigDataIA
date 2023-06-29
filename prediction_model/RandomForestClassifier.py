from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def random_forest_classifier(x_data_names=None, y_data_names=None):
    print("----------------------- Random Forest Classifier -----------------------")
    # Diviser les données en caractéristiques (X) et étiquettes (y)
    X = x_data_names
    y = y_data_names

    # On prend 20% des données pour le test
    test_size = 0.2
    print(f"Ratio de test : {test_size}")

    # Diviser les données en ensembles d'entraînement et de test (70% - 30%)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Créer le modèle Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Entraîner le modèle sur l'ensemble d'entraînement
    print("Entraînement du modèle...")
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

    print("----------------------- Fin du Random Forest Classifier -----------------------")
