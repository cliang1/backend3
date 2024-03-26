from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource # used for REST API building
from model.exercises import ExerciseModel

# Import the TitanicModel class from the model file
# from model.titanic import TitanicModel

exercise_api = Blueprint('exercise_api', __name__,
                   url_prefix='/api/exercise')

api = Api(exercise_api)

class ExerciseAPI:
    class _Predict(Resource):
        
        def post(self):
            """ Semantics: In HTTP, POST requests are used to send data to the server for processing.
            Sending passenger data to the server to get a prediction fits the semantics of a POST request.
            
            POST requests send data in the body of the request...
            1. which can handle much larger amounts of data and data types, than URL parameters
            2. using an HTTPS request, the data is encrypted, making it more secure
            3. a JSON formated body is easy to read and write between JavaScript and Python, great for Postman testing
            """     
            # Get the passenger data from the request
            person = request.get_json()

            # Get the singleton instance of the TitanicModel
            exerciseModel = ExerciseModel.get_instance()
            # Predict the survival probability of the passenger
            response = exerciseModel.predict(person)

            # Return the response as JSON
            return jsonify(response)

    api.add_resource(_Predict, '/predict')

# import json
# from flask import Blueprint, request
# from flask_restful import Api, Resource
# from model.exercises import Exercise
# exercise_api = Blueprint('exercise_api', __name__, url_prefix='/api/exercise')
# api = Api(exercise_api)

# class ExerciseAPI:
#     class Predict(Resource):
#         def post(self):
#             body = request.get_json()
#             print(body)
            
#             # Extracting data from the request body
#             diet = body.get('diet')
#             if diet is None:
#                 return {'message': f'Diet is missing'}, 400
            
#             time = body.get('time')
#             if time is None:
#                 return {'message': f'Time is missing'}, 400
            
#             kind = body.get('kind')
#             if kind is None:
#                 return {'message': f'Kind is missing'}, 400
            
#             # Additional data preprocessing or validation can be performed here
            
#             # Create a list containing the required information for prediction
#             info = [diet, time, kind]
            
#             # Use the Exercise class to make a prediction
#             result = Exercise.predict(info)
            
#             return {'message': result}

#     api.add_resource(Predict, '/predict')


