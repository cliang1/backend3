from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import seaborn as sns

class ExerciseModel:

    # a singleton instance of TitanicModel, created to train the model only once, while using it for prediction multiple times
    _instance = None
    
    # constructor, used to initialize the TitanicModel
    def __init__(self):
        # the titanic ML model
        self.model = None
        self.dt = None
        # define ML features and target
        self.features = ['id','diet', 'time', 'kind']
        self.target = 'pulse'
        # load the titanic dataset
        self.exercise_data = sns.load_dataset('exercise')
        # one-hot encoder used to encode 'embarked' column
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    # clean the titanic dataset, prepare it for training
    def _clean(self):
        # Drop unnecessary columns
        self.exercise_data.drop(['id'], axis=1, inplace=True)
        self.exercise_data.drop(['diet'], axis=1, inplace=True)

        # Convert boolean columns to integers
        self.exercise_data['time'] = self.exercise_data['time'].apply(lambda x: int(x.split()[0]))
        
        # One-hot encode 'embarked' column
        onehot = self.encoder.fit_transform(self.exercise_data[['kind']]).toarray()
        cols = ['kind_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.exercise_data = pd.concat([self.exercise_data, onehot_df], axis=1)
        self.exercise_data.drop(['kind'], axis=1, inplace=True)

        # Add the one-hot encoded 'embarked' features to the features list
        self.features.extend(cols)
        
        # Drop rows with missing values
        self.exercise_data.dropna(inplace=True)

    # train the titanic model, using logistic regression as key model, and decision tree to show feature importance
    def _train(self):
        # split the data into features and target
        X = self.exercise_data[self.features]
        y = self.exercise_data[self.target]
        
        # perform train-test split
        self.model = LogisticRegression(max_iter=1000)
        
        # train the model
        self.model.fit(X, y)
        
        # train a decision tree classifier
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)
        
    @classmethod
    def get_instance(cls):   
        # check for instance, if it doesn't exist, create it
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        # return the instance, to be used for prediction
        return cls._instance

    def predict(self, person):
        
        person_df = pd.DataFrame(person, index=[0])

        self.exercise_data['time'] = self.exercise_data['time'].apply(lambda x: int(x.split()[0]))

        onehot = self.encoder.fit_transform(self.exercise_data[['kind']]).toarray()
        cols = ['kind_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.exercise_data = pd.concat([self.exercise_data, onehot_df], axis=1)
        self.exercise_data.drop(['kind'], axis=1, inplace=True)
        self.exercise_data.drop(['id'], axis=1, inplace=True)
        self.exercise_data.drop(['diet'], axis=1, inplace=True)
        
        pulse = np.squeeze(self.model.predict_proba(person_df))

        return {'pulse': pulse}
    
    def feature_weights(self):
        importances = self.dt.feature_importances_
        return {feature: importance for feature, importance in zip(self.features, importances)} 
    
def initExercise():
    ExerciseModel.get_instance()
    
def testExercise():
    print(" Step 1:  Define theoritical passenger data for prediction: ")
    person = {
        'id': ['2'],
        'diet': ['low fat'],
        'time': ['1 min'],
        'kind': ['rest'],

    }
    print("\t", person)
    print()

    # get an instance of the cleaned and trained Titanic Model
    exerciseModel = ExerciseModel.get_instance()
    print(" Step 2:", exerciseModel.get_instance.__doc__)

    print(" Step 3:", exerciseModel.predict.__doc__)
    probability = exerciseModel.predict(person)
    print('\t pulse probability: {:.2%}'.format(probability.get('pulse')))
    print()
    
    # print the feature weights in the prediction model
    print(" Step 4:", exerciseModel.feature_weights.__doc__)
    importances = exerciseModel.feature_weights()
    for feature, importance in importances.items():
        print("\t\t", feature, f"{importance:.2%}") # importance of each feature, each key/value pair
        
if __name__ == "__main__":
    print(" Begin:", testExercise.__doc__)
    testExercise()