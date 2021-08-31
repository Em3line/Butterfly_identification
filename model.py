from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import (decode_predictions,
                                                          preprocess_input)

def load_model():
    # loads and returns teh pretrained model
    filepath = "/Users/prunelle/code/Em3line/Butterfly_identification/raw_data/models/API_FTW"
    model = load_keras_model(filepath, compile=True, options=None)
    return model

def prepare_image(image, target):
    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def predict(image, model):
    # We keep the 2 classes with the highest confidence score
    results = decode_predictions(model.predict(image), 2)[0]
    response = [
        {"class": result[1], "score": float(round(result[2], 3))} for result in results
    ]
    return response
