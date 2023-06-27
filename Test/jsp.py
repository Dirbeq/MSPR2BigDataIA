import pandas
import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def jsp(data_brut=None, x_data_names=None, y_data_names=None):
    # The goal is to predict collumn "Nom" based on the other collumns
    # So we remove the column "Nom" from the data and store it in a variable

    labels = data_brut[y_data_names]
    print("Labels loaded")

    # We create a model with 3 layers
    # The first layer is a normalization layer
    def build_and_compile_model(norm):
        model = keras.Sequential([
            norm,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        # We compile the model with the adam optimizer and the mean squared error as loss function
        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    # We create a normalization layer
    normalizer = preprocessing.Normalization()
    print("Normalization layer created")

    # Select data for training ,Code de la circonscription, % Abs/Ins
    data = data_brut[x_data_names]
    print("Data selected")

    # We adapt the normalization layer to the data
    normalizer.adapt(numpy.array(data))
    print("Normalization layer adapted")

    # We create the model
    dnn_model = build_and_compile_model(normalizer)
    print("Model created")

    # We train the model with 20 epochs
    history = dnn_model.fit(
        data, labels,
        validation_split=0.2,
        verbose=1, epochs=10)
    print("Model trained")

    # Save the model
    dnn_model.save("./model/model.h5")
