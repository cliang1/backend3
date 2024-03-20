#Importating Packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pdb
import pickle
from sklearn.metrics import accuracy_score

class Titanic():
    #Loading, pre-proccessing, and splitting of data
    def initialize():
        data_train = pd.read_csv('./train.csv')
        data_train['Age'].fillna(data_train['Age'].mean(), inplace=True)
        data_train['Sex'] = data_train['Sex'].map({'male': 1, 'female': 0})
        X = data_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
        y = data_train['Survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #Training Model (skipped b/c model already trained in file.)
        log_reg = LogisticRegression(solver='liblinear')
        #log_reg.fit(X_train, y_train)
        save_path = "logistic_regression_model.pkl"
        with open(save_path, 'wb') as model_file:
            pickle.dump(log_reg, model_file)

    #userInput=[pclass, sex, age, sibsp,fare]
    def predict(userInput):
        save_path = "logistic_regression_model.pkl"
        userInput = pd.DataFrame.from_dict([userInput])
        with open(save_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        survival = loaded_model.predict_proba(userInput)[0]
        print(survival)
        return survival[1]
    
