from flask import Blueprint, jsonify
from flask_restful import Api, Resource
import requests
import random

from model.exercises import ExerciseModel

exercise_api = Blueprint('exercise_api', __name__, url_prefix='/api/exercises')
api = Api(exercise_api)

class ExerciseAPI:
    class _Read(Resource):
        def get(self):
            instance = ExerciseModel.get_instance()
            return jsonify(instance.feature_weights())

    class _ReadRandom(Resource):
        def get(self):
            instance = ExerciseModel.get_instance()
            person = {
                "diet": random.choice(["low fat", "no fat"]),
                "time": random.choice(["1 min", "15 min", "30 min"]),
                "kind": random.choice(["rest", "walking", "running"])
            }
            return jsonify(instance.predict(person))

    class _Predict(Resource):
        def post(self, data):
            instance = ExerciseModel.get_instance()
            return jsonify(instance.predict(data))

    api.add_resource(_Read, '/')
    api.add_resource(_ReadRandom, '/random')
    api.add_resource(_Predict, '/predict')

if __name__ == "__main__":
    server = "http://127.0.0.1:5000"  # Change to your server address
    url = server + "/api/exercises"
    responses = []

    # Fetch feature weights
    responses.append(requests.get(url))

    # Fetch a random prediction
    responses.append(requests.get(url + "/random"))

    # Test prediction with custom data
    data = {"diet": "low fat", "time": "15 min", "kind": "rest"}
    responses.append(requests.post(url + "/predict", json=data))

    # Display responses
    for response in responses:
        print(response)
        try:
            print(response.json())
        except:
            print("unknown error")
