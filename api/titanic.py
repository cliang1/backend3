import json, jwt
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import Api, Resource # used for REST API building
from datetime import datetime
from auth_middleware import token_required
import pdb

from model.titanic import Titanic #imports titanic data

titanic_api = Blueprint('titanic_api', __name__,
                   url_prefix='/api/titanic')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(titanic_api)
class TitanicAPI:        
    class Predict(Resource): 
        def post(self):
            body = request.get_json()
            print(body)
            # Initializing all parameters
            pclass = body.get('Pclass')
            if pclass is None:
                return {'message': f'pclass is missing'}, 400
            age = body.get('Age')
            if age is None:
                return {'message': f'Age is missing'}, 400
            sex = body.get('Sex')
            if sex is None:
                return {'message': f'Sex is missing'}, 400
            fare = body.get('Fare')
            if fare is None:
                return {'message': f'Fare is missing'}, 400
            sibsp = body.get('SibSp')
            if sibsp is None:
                return {'message': f'sibsp is missing'}, 400
            parch = body.get('Parch')
            
            if parch is None:
                return {'message': f'parch is missing'}, 400
            
            info = [pclass,age,sex,sibsp,parch,fare]

            result = Titanic.predict(body)
            
            return { 'message': result }
        
    api.add_resource(Predict, '/predict')