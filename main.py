import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.svm import SVC

from perform import perform_on_model
from prediction_model.jsp import jsp

data = pd.read_csv('./data/data.csv', on_bad_lines='skip', sep=",", index_col=1)
x_data = data[['Médiane du niveau de vie 2020', 'Taux de chômage annuel moyen 2022', 'Taux de pauvreté 2020',
               'Coups et blessures volontaires (taux) 2022']]  # Data used to predict = Features
y_data = data['2022 Gagnant 1 tour']  # Predicted value = Label

# y_data_names contains value as string, we need to convert it to int
y_data = y_data.replace('EG', 0)
y_data = y_data.replace('G', 1)
y_data = y_data.replace('C', 2)
y_data = y_data.replace('D', 3)
y_data = y_data.replace('ED', 4)

if __name__ == '__main__':
    perform_on_model(x_data=x_data, y_data=y_data,
                     model=RandomForestClassifier(n_estimators=100, random_state=42))

    print("\n")

    perform_on_model(x_data=x_data, y_data=y_data, model=SVC(verbose=False, random_state=42, probability=True))

    print("\n")

    perform_on_model(x_data=x_data, y_data=y_data, model=LogisticRegression(max_iter=300))

    print("\n")

    perform_on_model(x_data=x_data, y_data=y_data, model=SGDRegressor(max_iter=100))

    jsp(x_data=x_data, y_data=y_data)

    # reso_de_neuron(x_data=x_data, y_data=y_data) //TODO: fix this, or/and do KNN?

    # # Anomaly detection via Isolation Forest
    # isolation_forest(x_data_names=x_data_names, y_data_names=y_data_names)
