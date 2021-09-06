from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# import pandas as pd
# import numpy as np
import pickle
# from tensorflow.keras.applications.imagenet_utils import decode_predictions
from app.model import load_model, predict, prepare_image, get_prediction_pictures
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI #, File, HTTPException, UploadFile
import matplotlib.pyplot as plt
import base64
from urllib.parse import unquote, quote
from PIL import Image
from starlette.requests import Request

# bloc ci-dessous décomenté lors du fonctionnement de l'api alexandre
model = load_model()


#Define the response JSON
class Prediction(BaseModel):
    filename: str
    content_type: str
    predictions: List[dict] = []


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"])  # Allows all headers


@app.get("/test-operation-bidon")
def test_operation_bidon(entered_data):
    return (f"{int(entered_data)*2} millions de papillons")


@app.get("/predict-image")
def predict_image(url):
    pkl_file = open(url, 'rb')
    image = pickle.load(pkl_file)
    im = plt.imread(image)
    image = prepare_image(im)
    prediction = predict(image, model)
    pkl_file.close()
    dico = {}
    for i in prediction.keys():
        nom_latin = i
        pkl_files = get_prediction_pictures(i)
        dico[prediction[i]] = (nom_latin, pkl_files)
    return dico


@app.post("/predict-image-str")
async def predict_image_str(request: Request):
    print('toto')
    data = await request.json()
    string = data['string']
    print(string)
    with open("imageToSave.jpg", "wb") as fh:
        fh.write(base64.decodebytes(bytes(string, 'utf-8')))
    im = plt.imread("imageToSave.jpg")
    image = prepare_image(im)
    prediction = predict(image, model)
    dico = {}
    for i in prediction.keys():
        nom_latin = i
        pkl_files = get_prediction_pictures(i)
        dico[prediction[i]] = (nom_latin, pkl_files)
    return dico
#from io import BytesIO, StringIO
#from PIL import Image
#code qui fonctionne (Alexandre)
# @app.post("/predict")
# async def prediction(file: UploadFile = File(...)):
#     print('before file check')
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="File provided is not an image.")
#     content = await file.read()
#     image = Image.open(BytesIO(content)).convert("RGB")
#     return "toto"

# @app.post("/predict")
# async def prediction(file: UploadFile = File(...)):
    # Initialize the data dictionnary that will be returned
    # print('before file check')
    # if not file.content_type.startswith("image/"):
    #     raise HTTPException(status_code=400, detail="File provided is not an image.")
    # img = Image.open(file.file).convert('RGB')
    # make prediction
    #image = Image.frombytes("RGB", (1000, 1000), content)
    #image = Image.open(BytesIO(content)).convert("RGB")
    #Faire la prédiction en utilisant l'image
    # image = prepare_image(content, target=(224, 224))
    # prediction = predict(image, model)
    # return the response as a JSON
    # return "toto"
    # return {
    #     "filename": file.filename,
    #     "content_type": file.content_type,
    #     "predictions": prediction,
    # }

# @app.post("/predict", response_model=Prediction)
# async def prediction(file: UploadFile = File(...)):
#     # Ensure that the file is an image
#     content = await file.read()
#     image = Image.open(BytesIO(content)).convert("RGB")
#     # preprocess the image and prepare it for classification
#     image = prepare_image(image, target=(224, 224))
#     response = predict(image, model)
#     # return the response as a JSON
#     return {
#         "filename": file.filename,
#         "content_type": file.content_type,
#         "predictions": response,
#     }
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=5000)

# @app.post("/predict-specy", response_model=Prediction)
# def predict_specy(url_image, model):
#     pkl_file = open(url_image, 'rb')
#     image_papillon = pickle.load(pkl_file)
#     pkl_file.close()
#     results = decode_predictions(model.predict(image_papillon), 2)[0]
#     response = [
#         {"class": result[1], "score": float(round(result[2], 3))} for result in results
#     ]
#     return response
