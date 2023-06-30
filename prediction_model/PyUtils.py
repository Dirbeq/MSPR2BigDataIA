import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def calculate_scores(y_true, y_pred, labels, model_name):
    print('Accuracy: {}'.format(round(accuracy_score(y_true, y_pred), 2)))
    print('Precision: {}'.format(round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 2)))
    print('Recall: {}'.format(round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 2)))
    print('F1 score: {}'.format(round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 2)))
    if model_name != "SGDRegressor":
        plot_confusion_matrix(y_true, y_pred, labels, model_name)


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
