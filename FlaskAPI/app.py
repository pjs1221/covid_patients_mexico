import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import pickle
import numpy as np



def load_models():
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

app = Flask(__name__)
@app.route('/predict', methods=['GET'])
def predict():
    #stub input features
    request_json = request.get_json()
    x = request_json['input']
    x_in = np.array(x).reshape(1,-1)
    #load model
    model = load_models()
    #Predict based of model and previously determined threshold
    threshold = 0.067
    prediction = model.predict_proba(x_in)[0][1]
    if(prediction >= threshold):
        prediction = 1 
    else:
        prediction = 0
    response = json.dumps({'response': prediction})
    return response, 200

if __name__ == '__main__':
    application.run(debug=True)