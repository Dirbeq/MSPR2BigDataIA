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
    verbose = False

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

    print("----------------------- Fin SVM -----------------------")
