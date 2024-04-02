from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.fitnesses import FitnessModel

fitness_api = Blueprint('fitness_api', __name__, url_prefix='/api/fitness')
api = Api(fitness_api)

class FitnessAPI:
    class _Predict(Resource):
        def post(self):
            """ Semantics: In HTTP, POST requests are used to send data to the server for processing.
            Sending exercise data to the server to get a prediction fits the semantics of a POST request.
            POST requests send data in the body of the request...
            1. which can handle much larger amounts of data and data types, than URL parameters
            2. using an HTTPS request, the data is encrypted, making it more secure
            3. a JSON formatted body is easy to read and write between JavaScript and Python, great for Postman testing
            Sample JSON data:
            {
                "Duration": 37,
                "BPM": 170,
                "Intensity": 5
            }
            """
            # Get the exercise data from the request
            exercise_data = request.get_json()
            # Get the singleton instance of the FitnessModel
            fitnessModel = FitnessModel.get_instance()
            # Predict the calorie burn based on exercise data
            predicted_calories = fitnessModel.predict(exercise_data)
            # Return the predicted calorie burn as JSON
            return jsonify({'predicted_calories': predicted_calories})

    api.add_resource(_Predict, '/predict')

def initFitnessModel():
    """ Initialize the Fitness Model.
    This function is used to load the Fitness Model into memory and prepare it for prediction.
    """
    FitnessModel.get_instance()

def testFitnessModel():
    """ Test the Fitness Model
    Using the FitnessModel class, we can predict the calorie burn based on exercise features.
    Print output of this test contains method documentation, exercise data, and predicted calorie burn.
    """
    # Setup exercise data for prediction
    print(" Step 1: Define exercise data for prediction: ")
    exercise_data = {
        "Duration": 37,
        "BPM": 170,
        "Intensity": 5
    }
    print("\t", exercise_data)
    print()
    # Get an instance of the cleaned and trained Fitness Model
    fitnessModel = FitnessModel.get_instance()
    print(" Step 2:", fitnessModel.get_instance.__doc__)
    # Print the predicted calorie burn
    print(" Step 3:", fitnessModel.predict.__doc__)
    predicted_calories = fitnessModel.predict(exercise_data)
    print('\t Predicted Calorie Burn:', predicted_calories)
    print()

if __name__ == "__main__":
    print(" Begin:", testFitnessModel.__doc__)
    testFitnessModel()
