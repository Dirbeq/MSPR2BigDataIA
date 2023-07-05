import numpy
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from PyUtils import data_preprocessing


def reseau_neurones_artificiels():
    # Read data from csv file
    data = pd.read_csv('./data/data.csv', on_bad_lines='skip', sep=",", index_col=1)

    x_data, y_data = data_preprocessing(data=data, preprocessing=False)

    # y_data_names contains value as string, we need to convert it to int
    y_data = y_data.replace('EG', 0)
    y_data = y_data.replace('G', 1)
    y_data = y_data.replace('C', 2)
    y_data = y_data.replace('D', 3)
    y_data = y_data.replace('ED', 4)

    print("----------------------- réseau de neurones artificiels  -----------------------")
    # réseau de neurones artificiels (multi-layer perceptron) avec normalisation des données
    # et optimisation par la rétropropagation du gradient (backpropagation).

    labels = y_data
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

    # Select data for training
    data = x_data
    print("Data selected")

    # We adapt the normalization layer to the data
    normalizer.adapt(numpy.array(data))
    print("Normalization layer adapted")

    # We create the model
    dnn_model = build_and_compile_model(normalizer)
    print("Model created")

    # We train the model with 20 epochs
    print("Training model...")
    dnn_model.fit(
        data, labels,
        validation_split=0.2,
        verbose=0, epochs=500)
    print("Model trained")

    # Evaluate the model
    results = dnn_model.evaluate(data, labels, verbose=0)
    print("Results :", results)

    # Save the model
    dnn_model.save("./model/model.h5")

    print(" ----------------------- Fin réseau de neurones artificiels  -----------------------")
