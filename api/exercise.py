from flask import Blueprint, request, jsonify, Flask
from flask_restful import Api, Resource # used for REST API building
from model.exercises import ExerciseModel

app = Flask(__name__)

# Initialize ExerciseModel instance and train the models
exercise_model = ExerciseModel.get_instance()
exercise_model.init_exercise_list()
exercise_model._clean()
exercise_model._train()  # Call _train method to train the models

@app.route('/predict', methods=['POST'])
def post():
    data = request.get_json()
    person = data['person']
    response = exercise_model.predict(person)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
