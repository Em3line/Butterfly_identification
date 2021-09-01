from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.applications.imagenet_utils import (decode_predictions)
from model import load_model, predict, prepare_image
from pydantic import BaseModel
from typing import List
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
import uvicorn

# model = load_model()

# #Define the response JSON
# class Prediction(BaseModel):
#     filename: str
#     content_type: str
#     predictions: List[dict] = []
# app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/test-operation-bidon")
def test_operation_bidon(entered_data):
    return (f"{int(entered_data)*2} millions de papillons")

# @app.get("/predict-image")
# def predict_image(url):
#     pkl_file = open(url, 'rb')
#     image = pickle.load(pkl_file)
#     im = Image.open(url)
#     im.show()
#     pkl_file.close()
#     return ("image ok")

#code qui fonctionne (Alexandre)
@app.post("/predict")
async def prediction(file: UploadFile = File(...)):
    # Initialize the data dictionnary that will be returned
    print('before file check')
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    content = await file.read()
    image = Image.open(BytesIO(content)).convert("RGB")
    #Faire la prédiction en utilisant l'image
    return "toto"

# @app.post("/predict")
# async def prediction(file: UploadFile = File(...)):
#     # Initialize the data dictionnary that will be returned
#     print('before file check')
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="File provided is not an image.")
#     content = await file.read()
#     image = Image.open(BytesIO(content)).convert("RGB")
#     #Faire la prédiction en utilisant l'image
#     image = prepare_image(image, target=(224, 224))
#     response = predict(image, model)
#     # return the response as a JSON
#     return {
#         "filename": file.filename,
#         "content_type": file.content_type,
#         "predictions": response,
#     }

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
