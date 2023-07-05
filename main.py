import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from PyUtils import data_preprocessing, perform_on_model
from prediction_model.RNA import reseau_neurones_artificiels

# Read data from csv file
data = pd.read_csv('./data/data.csv', on_bad_lines='skip', sep=",", index_col=1)

x_data, y_data = data_preprocessing(data=data)

if __name__ == '__main__':
    perform_on_model(x_data=x_data, y_data=y_data,
                     model=RandomForestClassifier(n_estimators=100, random_state=42))

    print("\n")

    perform_on_model(x_data=x_data, y_data=y_data,
                     model=SVC(verbose=False, random_state=42, probability=True))

    print("\n")

    perform_on_model(x_data=x_data, y_data=y_data,
                     model=LogisticRegression(max_iter=300))

    print("\n")

    reseau_neurones_artificiels()

    # # Anomaly detection via Isolation Forest
    # isolation_forest(x_data_names=x_data_names, y_data_names=y_data_names)
