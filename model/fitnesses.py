## Python Titanic Model, prepared for a titanic.py file
# Import the required libraries for the TitanicModel class
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class FitnessModel:
    """A class used to represent the Calorie Burn Model based on features like Duration, BPM, and Intensity.
    """
    # a singleton instance of CalorieModel, created to train the model only once, while using it for prediction multiple times
    _instance = None
    # constructor, used to initialize the CalorieModel
    def __init__(self):
        # the calorie burn model
        self.model = None
        self.dt = None
        # define ML features and target
        self.features = ['Duration', 'BPM', 'Intensity']
        self.target = 'Calories'
        # load the calorie burn dataset
        self.fitness_data = pd.read_csv('fitness.csv')  # Assuming the data file is 'calorie_burn_data.csv'
    # clean the calorie burn dataset, prepare it for training
    def _clean(self):
        # Drop rows with missing values
        self.fitness_data.dropna(inplace=True)
    # train the calorie burn model, using linear regression as key model, and decision tree to show feature importance
    def _train(self):
        # split the data into features and target
        X = self.fitness_data[self.features]
        y = self.fitness_data[self.target]
        # perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # train the model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        # train a decision tree regressor
        self.dt = DecisionTreeRegressor()
        self.dt.fit(X_train, y_train)
    @classmethod
    def get_instance(cls):
        """ Gets, and conditionally cleans and builds, the singleton instance of the CalorieModel.
        The model is used for predicting calorie burn based on duration, BPM, and intensity.
        Returns:
            CalorieModel: the singleton _instance of the CalorieModel, which contains data and methods for prediction.
        """
        # check for instance, if it doesn't exist, create it
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        # return the instance, to be used for prediction
        return cls._instance
    def predict(self, data_point):
        """ Predict the calorie burn for a given data point.
        Args:
            data_point (dict): A dictionary representing a data point. The dictionary should contain the following keys:
                'Duration': The duration of exercise
                'BPM': Beats per minute
                'Intensity': Intensity level of exercise
        Returns:
           float: Predicted calorie burn
        """
        # clean the data point
        data_point_df = pd.DataFrame(data_point, index=[0])
        # predict the calorie burn
        calorie_burn = self.model.predict(data_point_df[self.features])[0]
        # return the predicted calorie burn
        return calorie_burn
    def feature_weights(self):
        """Get the feature weights
        The weights represent the relative importance of each feature in the prediction model.
        Returns:
            dictionary: contains each feature as a key and its weight of importance as a value
        """
        # extract the feature importances from the decision tree model
        importances = self.dt.feature_importances_
        # return the feature importances as a dictionary, using dictionary comprehension
        return {feature: importance for feature, importance in zip(self.features, importances)}

def initFitnessModel():
    """ Initialize the Calorie Model.
    This function is used to load the Calorie Model into memory, and prepare it for prediction.
    """
    FitnessModel.get_instance()

def testFitnessModel():
    """ Test the Calorie Model
    Using the CalorieModel class, we can predict the calorie burn based on exercise features.
    Print output of this test contains method documentation, data point, and predicted calorie burn.
    """
    # setup data point for prediction
    print(" Step 1: Define data point for prediction: ")
    data_point = {
        'Duration': [37],
        'BPM': [170],
        'Intensity': [5]
    }
    print("\t", data_point)
    print()
    # get an instance of the cleaned and trained Calorie Model
    fitnessModel = FitnessModel.get_instance()
    print(" Step 2:", fitnessModel.get_instance.__doc__)
    # print the predicted calorie burn
    print(" Step 3:", fitnessModel.predict.__doc__)
    predicted_calories = fitnessModel.predict(data_point)
    print('\t Predicted Calorie Burn:', predicted_calories)
    print()
    # print the feature weights in the prediction model
    print(" Step 4:", fitnessModel.feature_weights.__doc__)
    importances = fitnessModel.feature_weights()
    for feature, importance in importances.items():
        print("\t\t", feature, f"{importance:.2%}") # importance of each feature, each key/value pair

if __name__ == "__main__":
    print(" Begin:", testFitnessModel.__doc__)
    testFitnessModel()