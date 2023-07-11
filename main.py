import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from PyUtils import data_preprocessing, perform_on_model

# Read data from csv file
data = pd.read_csv('./data/data_departement.csv', on_bad_lines='skip', sep=",", index_col=1)

# Perform data preprocessing
x_data, y_data = data_preprocessing(data=data)

if __name__ == '__main__':
    # Perform evaluation on Random Forest Classifier
    perform_on_model(x_data=x_data, y_data=y_data,
                     model=RandomForestClassifier(n_estimators=100, random_state=None))

    print("\n")

    # Perform evaluation on Support Vector Classifier
    perform_on_model(x_data=x_data, y_data=y_data,
                     model=SVC(verbose=False, random_state=None, probability=True))

    print("\n")

    # Perform evaluation on Logistic Regression
    perform_on_model(x_data=x_data, y_data=y_data,
                     model=LogisticRegression(max_iter=400))

    print("\n")

    # Perform evaluation on Linear Discriminant Analysis
    perform_on_model(x_data=x_data, y_data=y_data,
                        model=LinearDiscriminantAnalysis())

    print("\n")

    # Perform evaluation on K-Nearest Neighbors
    perform_on_model(x_data=x_data, y_data=y_data,
                        model=KNeighborsClassifier(n_neighbors=5))


    # # Anomaly detection via Isolation Forest
    # isolation_forest(x_data_names=x_data_names, y_data_names=y_data_names)
