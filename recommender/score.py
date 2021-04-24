import json
import numpy as np
import os
import joblib
import logging
from azureml.core.model import Model
from non_personalized import *

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'local_model.pkl')
    model = joblib.load(model_path)

def run(request):
    try:
        data = json.loads(request)
        user_id = data['data']
        result = model.predict(user_id, 5)
        return {'data' : result.tolist() , 'message' : "Successfully recommended an item"}
    except Exception as e:
        error = str(e)
        return {'data' : request , 'message' : 'Failed to recommend an item'}
    