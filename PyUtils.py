import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def perform_on_model(model, x_data, y_data, test_size=0.2):
    X = x_data
    y = y_data

    model_name = type(model).__name__

    print(f"----------------------- {model_name} -----------------------")
    print(f"Ratio de test : {test_size}")

    # Diviser les données en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Fit the label encoder on the entire dataset
    label_encoder = LabelEncoder()
    label_encoder.fit(y_data)

    # Transform labels for training and test sets
    y_train_encoded = label_encoder.transform(y_train)

    # Entraîner le modèle sur l'ensemble d'entraînement
    print("Entraînement du modèle...")
    model.fit(x_train, y_train_encoded)

    # Prédire les étiquettes sur l'ensemble d'entraînement
    y_pred_train = model.predict(x_train)

    # Calculer les scores sur l'ensemble d'entraînement
    print("Calcul des scores sur l'ensemble d'entraînement...")
    calculate_scores(y_train, y_pred_train, label_encoder, model_name=model_name)

    # Prédire les étiquettes sur l'ensemble de test
    y_pred_test = model.predict(x_test)

    # Calculer les scores sur l'ensemble de test
    print("Calcul des scores sur l'ensemble de test...")
    calculate_scores(y_test, y_pred_test, label_encoder, model_name=model_name)

    print(f"----------------------- Fin {model_name} -----------------------")


def calculate_scores(y_true, y_pred, label_encoder, model_name):
    # Inverse transform the encoded labels to original labels
    y_true = label_encoder.inverse_transform(y_true)
    y_pred = label_encoder.inverse_transform(y_pred)

    print('Accuracy: {}'.format(round(accuracy_score(y_true, y_pred), 2)))
    print('Precision: {}'.format(round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 2)))
    print('Recall: {}'.format(round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 2)))
    print('F1 score: {}'.format(round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 2)))

    if model_name != "SGDRegressor":
        plot_confusion_matrix(y_true, y_pred, label_encoder.classes_, model_name)


def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    cm = confusion_matrix(y_true, y_pred)
    _, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Confusion matrix for " + model_name)
    plt.show()

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    _, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix for " + model_name)
    plt.show()


def data_preprocessing(data, preprocessing=True):
    # Check for missing values
    print(data.isnull().sum())

    # Extract features and labels
    x_data = data[['Médiane du niveau de vie 2020', 'Taux de chômage annuel moyen 2022', 'Taux de pauvreté 2020',
                   'Coups et blessures volontaires (taux) 2022']]
    y_data = data['2022 Gagnant 1 tour']

    if preprocessing:
        # Encode categorical labels
        label_encoder = LabelEncoder()
        y_data = label_encoder.fit_transform(y_data)

        # Perform outlier detection and handling
        # You can use appropriate techniques like IQR or Z-score to identify and handle outliers

        # Perform feature scaling or normalization
        # scaler = StandardScaler()
        # x_data = scaler.fit_transform(x_data)

    return x_data, y_data
