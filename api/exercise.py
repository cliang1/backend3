

import json
from flask import Blueprint, request
from flask_restful import Api, Resource
from model.exercises import Exercise
exercise_api = Blueprint('exercise_api', __name__, url_prefix='/api/exercise')
api = Api(exercise_api)

class ExerciseAPI:
    class Predict(Resource):
        def post(self):
            body = request.get_json()
            print(body)
            
            # Extracting data from the request body
            diet = body.get('diet')
            if diet is None:
                return {'message': f'Diet is missing'}, 400
            
            time = body.get('time')
            if time is None:
                return {'message': f'Time is missing'}, 400
            
            kind = body.get('kind')
            if kind is None:
                return {'message': f'Kind is missing'}, 400
            
            # Additional data preprocessing or validation can be performed here
            
            # Create a list containing the required information for prediction
            info = [diet, time, kind]
            
            # Use the Exercise class to make a prediction
            result = Exercise.predict(info)
            
            return {'message': result}

    api.add_resource(Predict, '/predict')


