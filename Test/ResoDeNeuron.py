import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

# Charger les données à partir du fichier CSV
data = pd.read_csv('../data/Data1.csv',on_bad_lines='skip', sep=";", index_col=0)

# Diviser les données en caractéristiques (X) et étiquettes (y)
X = data[['Code de la circonscription', '% Abs/Ins']]
y = data['N°Panneau']

# Normaliser les caractéristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Créer le modèle de réseau de neurones
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(2,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compiler le modèle
model.compile(optimizer=Adam(), loss='mse', metrics=[RootMeanSquaredError()])

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train, y_train, epochs=10, verbose=1)

# Évaluer le modèle sur les données de test
loss, rmse = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {loss}, RMSE: {rmse}")

# Prédire de nouvelles valeurs
new_data = pd.DataFrame([[26080, 12.2], [75078, 23.8]], columns=['Code de la circonscription', '% Abs/Ins'])
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
print(f"Prédictions : {predictions.flatten()}")
