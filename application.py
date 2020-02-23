from flask import Flask, request
from flask_restful import Resource, Api
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import numpy as np
import keras.backend.tensorflow_backend as tb

app = Flask(__name__)
api = Api(app)
model = None

def load_model():
    global model
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    print(model_from_json)
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

    def post(self):
        tb._SYMBOLIC_SCOPE.value = True
        file = request.files['file']
        print(file)
        return {'status': 'success'}


class Train(Resource):
    custom_vision_api_key = '938799f332bb4135b771e9e450a62828'
    end_point = 'https://burn-predictor.cognitiveservices.azure.com/'

    def get(self):
    
        return {'status': 'success'}


api.add_resource(Predict, '/predict')
api.add_resource(Train, '/train')

if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(debug=True, threaded=False)
