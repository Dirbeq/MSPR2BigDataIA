import pandas as pd

from Test.IsolationForest import isolation_forest
from Test.LogisticRegression import logistic_regression
from Test.RegressionLineaire import regression_lineaire
from Test.SVM import svm
from Test.jsp import jsp

data = pd.read_csv('./data/data.csv', on_bad_lines='skip', sep=",", index_col=1)
x_data_names = data[['Médiane du niveau de vie 2020', 'Taux de chômage annuel moyen 2022', 'Taux de pauvreté 2020',
                     'Coups et blessures volontaires (taux) 2022']]  # Data used to predict = Features
y_data_names = data['2022 Gagnant 1 tour']  # Predicted value = Label

# y_data_names contains value as string, we need to convert it to int
y_data_names = y_data_names.replace('EG', 0)
y_data_names = y_data_names.replace('G', 1)
y_data_names = y_data_names.replace('C', 2)
y_data_names = y_data_names.replace('D', 3)
y_data_names = y_data_names.replace('ED', 4)

if __name__ == '__main__':
    isolation_forest(x_data_names=x_data_names, y_data_names=y_data_names)
    logistic_regression(x_data_names=x_data_names, y_data_names=y_data_names)
    regression_lineaire(x_data_names=x_data_names, y_data_names=y_data_names)
    svm(x_data_names=x_data_names, y_data_names=y_data_names)
    jsp(x_data_names=x_data_names, y_data_names=y_data_names)
