from Test.IsolationForest import isolation_forest
from Test.RegressionLineaire import regression_lineaire
from Test.test1 import logistic_regression
from Test.SVM import svm
from Test.jsp import jsp
import pandas as pd

data = pd.read_csv('../data/Data1.csv', on_bad_lines='skip', sep=";", index_col=0)
x_data_names = ['Code de la circonscription', '% Abs/Ins'] # Data used to predict = Features
y_data_names = ['% Abs/Exp'] # Predicted value = Label

if __name__ == '__main__':
    isolation_forest(data, x_data_names, y_data_names)
    logistic_regression(data, x_data_names, y_data_names)
    regression_lineaire(data, x_data_names, y_data_names)
    svm(data, x_data_names, y_data_names)
    jsp(data, x_data_names, y_data_names)
