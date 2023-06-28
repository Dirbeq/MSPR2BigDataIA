import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def reso_de_neuron(x_data_names=None, y_data_names=None):
    print("----------------------- Réseau de neurones -----------------------")
    # Diviser les données en caractéristiques (X) et étiquettes (y)
    X = x_data_names
    y = y_data_names

    # Normaliser les caractéristiques
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    # Diviser les données en jeux d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # Créer le modèle de réseau de neurones
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(2,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    # Compiler le modèle
    model.compile(optimizer=Adam(), loss='mse', metrics=[RootMeanSquaredError()])

    # Entraîner le modèle sur les données d'entraînement
    model.fit(x_train, y_train, epochs=10, verbose=1)

    # Évaluer le modèle sur les données de test
    loss, rmse = model.evaluate(x_test, y_test, verbose=0)
    print(f"Loss: {loss}, RMSE: {rmse}")

    # Accuracy
    accuracy = model.score(x_test, y_test)
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

    print("----------------------- Fin réseau de neurones -----------------------")
