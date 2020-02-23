from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
import requests

app = Flask(__name__)
api = Api(app)
model = None
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


class PredictAzure(Resource):
    custom_vision_api_key = '938799f332bb4135b771e9e450a62828'
    end_point = "https://burn-predictor.cognitiveservices.azure.com/customvision/v3.0/Prediction/cdf629fa-a506-47d7-aaf4-856341633206/classify/iterations/Iteration1/image"

    @cross_origin()
    def post(self):
        file = request.files['file']
        filepath = file.filename
        file.save(filepath)

        response = requests.post(self.end_point, data=open(filepath, 'rb').read(),
                                 headers={'Prediction-Key': '6b516e98b52b48ec8154f5ce922accb4',
                                          'Content-Type': 'application/octet-stream'})
        print(response.json())

        predictions = response.json()['predictions']
        prediction = ""
        for pred in predictions:
            if pred['probability'] > 0.5:
                prediction = pred["tagName"]

        if prediction == "1":
            prediction = "1st Degree"
        elif prediction == "3":
            prediction = "3rd Degree"

        print("Sending: " + prediction)
        return {'prediction': prediction}


api.add_resource(PredictAzure, '/predict')

print(("* Flask starting server..."
       "please wait until server has fully started"))
app.run(host="0.0.0.0", debug=True, threaded=False)
