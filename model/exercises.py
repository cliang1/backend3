import os
import csv

from __init__ import app, db
from sqlalchemy.exc import IntegrityError

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import seaborn as sns

class ExerciseModel:

    _instance = None
    
    def __init__(self):
        self.model = None
        self.dt = None
        self.features = ['id', 'diet', 'time', 'kind']
        self.target = 'pulse'
        self.exercise_data = sns.load_dataset('exercise')
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def _clean(self):
        self.exercise_data.drop(['id'], axis=1, inplace=True)
        self.exercise_data.drop(['diet'], axis=1, inplace=True)
        self.exercise_data['time'] = self.exercise_data['time'].apply(lambda x: int(x.split()[0]))
        onehot = self.encoder.fit_transform(self.exercise_data[['kind']]).toarray()
        cols = ['kind_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.exercise_data = pd.concat([self.exercise_data, onehot_df], axis=1)
        self.features.extend(cols)
        self.exercise_data.drop(['kind'], axis=1, inplace=True)
        self.exercise_data.dropna(inplace=True)

    def _train(self):
        X = self.exercise_data[self.features]
        y = self.exercise_data[self.target]
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)
        
    @classmethod
    def get_instance(cls):   
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        return cls._instance

    def predict(self, person):
        person_df = pd.DataFrame(person, index=[0])
        pulse = np.squeeze(self.model.predict_proba(person_df))
        return {'pulse': pulse}
    
    def feature_weights(self):
        importances = self.dt.feature_importances_
        return {feature: importance for feature, importance in zip(self.features, importances)} 


class Exercise(db.Model):
    __tablename__ = 'exercises'

    id = db.Column(db.Integer, primary_key=True)
    diet = db.Column(db.String(50), nullable=False)
    pulse = db.Column(db.Integer, nullable=False)
    time = db.Column(db.String(50), nullable=False)
    kind = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

    def __init__(self, diet, pulse, time, kind, user_id):
        self.diet = diet
        self.pulse = pulse
        self.time = time
        self.kind = kind
        self.user_id = user_id

    def create(self):
        try:
            db.session.add(self)
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    def read(self):
        return {
            "id": self.id,
            "diet": self.diet,
            "pulse": self.pulse,
            "time": self.time,
            "kind": self.kind,
            "user_id": self.user_id
        }

def populate_database(csv_file):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Extract data from the CSV row
            user_data = {
                'name': row['name'],
                'uid': row['uid'],
                # Convert the date string to a datetime object
                'dob': datetime.strptime(row['dob'], '%Y-%m-%d').date()
            }
            post_data = {
                'note': row['note'],
                'image': row['image']
            }
            
            # Create a User instance
            user = User(**user_data)
            # Create a Post instance related to the User
            post = Post(**post_data, user=user)
            
            # Add User and Post instances to the session
            db.session.add(user)
            db.session.add(post)
        
        # Commit the changes to the database
        db.session.commit()

def initExercises():
    with app.app_context():
        db.create_all()

        csv_file_path = os.path.join(os.path.dirname(__file__), 'exercise.csv')
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                user_id = int(row['user_id'])
                exercise = Exercise(
                    diet=row['diet'],
                    pulse=int(row['pulse']),
                    time=row['time'],
                    kind=row['kind'],
                    user_id=user_id
                )
                exercise.create()


if __name__ == "__main__":
    # Path to the exercise.csv file
    csv_file_path = 'exercise.csv'
    # Initialize the database
    db.create_all()
    # Populate the database with data from the CSV file
    populate_database(csv_file_path)
    print("Database populated successfully.")