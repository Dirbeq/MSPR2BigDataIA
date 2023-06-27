import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def logistic_regression(data=None, x_data_names=None, y_data_names=None):
    # Diviser les données en caractéristiques (X) et étiquettes (y)
    X = data[x_data_names]
    y = data[y_data_names]

    # Diviser les données en jeux d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer le modèle
    model = LogisticRegression()

    # Spécifier le nombre d'époques
    num_epochs = 10
    print(f"Nombre d'époques : {num_epochs}")

    # Entraîner le modèle sur les données d'entraînement avec le nombre d'époques spécifié
    for epoch in tqdm(range(num_epochs)):
        model.fit(X_train, y_train)

    # Évaluer la précision du modèle sur les données de test
    accuracy = model.score(X_test, y_test)
    print(f"Précision du modèle : {accuracy}")

    # Prédire de nouvelles valeurs
    new_data = pd.DataFrame([[26080, 12.2], [75078, 23.8]], columns=['Code de la circonscription', '% Abs/Ins'])
    predictions = model.predict(new_data)
    print(f"Prédictions : {predictions}")
