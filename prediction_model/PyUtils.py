from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_scores(x, y):
    # Calculer la matrice de confusion
    confusion_matrix = confusion_matrix(x, y)
    print(f"Matrice de confusion : \n{confusion_matrix}")

    # Calculer la précision
    precision = precision_score(x, y)
    print(f"Précision : {precision}")

    # Calculer le rappel
    recall = recall_score(x, y)
    print(f"Rappel : {recall}")

    # Calculer le score F1
    f1 = f1_score(x, y)
    print(f"Score F1 : {f1}")

    # Calculer l'accuracy
    accuracy = accuracy_score(x, y)
    print(f"Accuracy : {accuracy}")
