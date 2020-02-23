from flask import Flask, request
from flask_restful import Resource, Api
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import numpy as np
import keras.backend.tensorflow_backend as tb
from flask_cors import CORS, cross_origin
from PIL import Image
import requests

app = Flask(__name__)
api = Api(app)
model = None
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def load_model():
    global model
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    json_file.close()

    # load weights into new model
    model.load_weights("first_try.h5")
    print("Loaded model from disk")
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


def prepare_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


class Predict(Resource):
    @cross_origin()
    def get(self):
        tb._SYMBOLIC_SCOPE.value = True
        prediction = ""

        image = load_img('tiger.png', target_size=(300, 600))
        result = model.predict(prepare_image(image))

        if result[0][0] == 1:
            prediction = "3rd Degree"
        else:
            prediction = "1st Degree"
        return {'prediction': prediction}

    @cross_origin()
    def post(self):
        tb._SYMBOLIC_SCOPE.value = True
        file = request.files['file']
        img = Image.open(file)

        result = model.predict(prepare_image(img))

        if result[0][0] == 1:
            prediction = "3rd Degree"
        else:
            prediction = "1st Degree"

        print("Sending: " + prediction)
        return {'prediction': prediction}


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


api.add_resource(Predict, '/predict')
api.add_resource(PredictAzure, '/predict-azure')

print(("* Loading Keras model and Flask starting server..."
       "please wait until server has fully started"))
load_model()
app.run(host="0.0.0.0", debug=True, threaded=False)
