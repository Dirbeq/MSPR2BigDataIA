from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def reso_de_neuron(x_data=None, y_data=None):
    print("----------------------- Réseau de neurones -----------------------")
    # Diviser les données en caractéristiques (X) et étiquettes (y)
    X = x_data
    y = y_data

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
    print("Entraînement du modèle...")
    model.fit(x_train, y_train, epochs=10, verbose=0)

    # Évaluer le modèle sur les données de test
    loss, rmse = model.evaluate(x_test, y_test, verbose=0)
    print(f"Loss: {loss}, RMSE: {rmse}")

    print("----------------------- Fin réseau de neurones -----------------------")
