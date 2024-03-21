import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

class Exercise:
    # Preprocessing and training the model
    @staticmethod
    def initialize():
        # Load exercise dataset
        exercise_data = pd.read_csv('./exercise.csv')

        # Preprocessing
        # Drop rows with missing values
        exercise_data.dropna(inplace=True)
        
        # Split the data into features (X) and target (y)
        X = exercise_data[['id', 'diet', 'time', 'kind']]
        y = exercise_data['pulse']
        
        # Perform one-hot encoding for the categorical columns
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['diet', 'time', 'kind'])], remainder='passthrough')
        X_encoded = ct.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

        # Train a linear regression model
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        
        # Save the trained model
        save_path = "exercise_linear_regression_model.pkl"
        with open(save_path, 'wb') as model_file:
            pickle.dump(regressor, model_file)

    # Predicting pulse rate
    @staticmethod
    def predict(user_input):
        # Load the trained model
        save_path = "exercise_linear_regression_model.pkl"
        with open(save_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        
        # Preprocess the user input
        user_input_df = pd.DataFrame.from_dict([user_input])
        
        # Perform one-hot encoding for the categorical columns
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['diet', 'time', 'kind'])], remainder='passthrough')
        user_input_encoded = ct.transform(user_input_df)

        # Predict the pulse rate
        pulse_prediction = loaded_model.predict(user_input_encoded)
        
        return pulse_prediction[0]

# Example usage for initializing the model and making predictions
# Uncomment the line below to initialize the model (execute it only once)
# Exercise.initialize()

# Example prediction
# userInput = {'id': 1, 'diet': 'low fat', 'time': '1 min', 'kind': 'rest'}
# predicted_pulse = Exercise.predict(userInput)
# print("Predicted Pulse Rate:", predicted_pulse)
