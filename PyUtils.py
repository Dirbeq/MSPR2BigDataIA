# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Function to perform evaluation on a machine learning model
def perform_on_model(model, x_data, y_data, test_size=0.2):
    # Split the data into features (x_data) and labels (y_data)
    X = x_data
    y = y_data

    # Get the name of the model for display purposes
    model_name = type(model).__name__

    # Print the test size ratio
    print(f"----------------------- {model_name} -----------------------")
    print(f"Test size ratio: {test_size}")

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Fit the label encoder on the entire dataset to encode categorical labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_data)

    # Transform labels for training and test sets
    y_train_encoded = label_encoder.transform(y_train)

    # Scale the data using StandardScaler
    print("Training the model...")
    model.fit(x_train, y_train_encoded)

    # Predict labels on the training set
    y_pred_train = model.predict(x_train)

    # Calculate evaluation scores on the training set
    print("Calculating scores on the training set...")
    calculate_scores(y_train, y_pred_train, label_encoder, model_name=model_name)

    # Predict labels on the test set
    y_pred_test = model.predict(x_test)

    # Calculate evaluation scores on the test set
    print("Calculating scores on the test set...")
    calculate_scores(y_test, y_pred_test, label_encoder, model_name=model_name)

    print(f"----------------------- End {model_name} -----------------------")


# Function to calculate evaluation scores (accuracy, precision, recall, F1 score)
def calculate_scores(y_true, y_pred, label_encoder, model_name):
    print('Accuracy: {}'.format(round(accuracy_score(y_true, y_pred), 2)))
    print('Precision: {}'.format(round(precision_score(y_true, y_pred, average='weighted'), 2)))
    print('Recall: {}'.format(round(recall_score(y_true, y_pred, average='weighted'), 2)))
    print('F1 score: {}'.format(round(f1_score(y_true, y_pred, average='weighted'), 2)))

    # Plot the confusion
    plot_confusion_matrix(y_true, y_pred, label_encoder.classes_, model_name)


# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a figure and axis for the confusion matrix plot
    _, ax = plt.subplots(figsize=(4, 4))

    # Display the confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)

    # Set the title for the confusion matrix plot
    plt.title("Confusion matrix for " + model_name)

    # Show the plot
    plt.show()

    # Calculate the normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    # Create a figure and axis for the normalized confusion matrix plot
    _, ax = plt.subplots(figsize=(4, 4))

    # Display the normalized confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)

    # Set the title for the normalized confusion matrix plot
    plt.title("Normalized confusion matrix for " + model_name)

    # Show the plot
    plt.show()


# Function for data preprocessing
def data_preprocessing(data, preprocessing=True):
    # Check for missing values in the dataset
    print(data.isnull().sum())

    # Extract features and labels from the data
    x_data = data[['Médiane du niveau de vie 2020', 'Taux de chômage annuel moyen 2022', 'Taux de pauvreté 2020',
                   'Coups et blessures volontaires (taux) 2022']]
    y_data = data['2022 Gagnant 1 tour']

    if preprocessing:
        # Encode categorical labels using LabelEncoder
        label_encoder = LabelEncoder()
        y_data = label_encoder.fit_transform(y_data)

        # Print the encoded labels
        print("Encoded labels:")
        print(label_encoder.classes_, '->', label_encoder.transform(label_encoder.classes_), '\n')

        # Perform outlier detection and handling (TODO: this part is not implemented yet)

        # Perform feature scaling or normalization using StandardScaler
        scaler = StandardScaler()
        x_data = scaler.fit_transform(x_data)

    return x_data, y_data
