import numpy
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# If pycharm marks the keras import as an error, do not worry, it is not.
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from PyUtils import data_preprocessing


def reseau_neurones_artificiels():
    # Read data from csv file
    data = pd.read_csv('./data/data_departement.csv', on_bad_lines='skip', sep=",", index_col=1)

    # Perform data preprocessing without additional preprocessing steps
    x_data, y_data = data_preprocessing(data=data, preprocessing=False)

    # Convert string labels to integers
    y_data = y_data.replace('EG', 0)
    y_data = y_data.replace('G', 1)
    y_data = y_data.replace('C', 2)
    y_data = y_data.replace('D', 3)
    y_data = y_data.replace('ED', 4)

    print("----------------------- Artificial Neural Network -----------------------")
    # Artificial Neural Network (multi-layer perceptron) with data normalization
    # and optimization using backpropagation.

    labels = y_data
    print("Labels loaded")

    # Function to build and compile the model with normalization layer
    def build_and_compile_model(norm):
        model = keras.Sequential([
            norm,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        # Compile the model with Adam optimizer and mean absolute error as the loss function
        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    # Create a normalization layer
    normalizer = preprocessing.Normalization()
    print("Normalization layer created")

    # Select data for training
    data = x_data
    print("Data selected")

    # Adapt the normalization layer to the data
    normalizer.adapt(numpy.array(data))
    print("Normalization layer adapted")

    # Create the model
    dnn_model = build_and_compile_model(normalizer)
    print("Model created")

    # Train the model with 500 epochs
    print("Training model...")
    dnn_model.fit(
        data, labels,
        validation_split=0.2,
        verbose=0, epochs=500)
    print("Model trained")

    # Evaluate the model
    results = dnn_model.evaluate(data, labels, verbose=0)
    print("Results:", results)

    # Save the model
    dnn_model.save("./model/model.h5")

    print("----------------------- End of Artificial Neural Network -----------------------")
