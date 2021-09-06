from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
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
    data = await request.json()
    string = data['string']
    with open("imageToSave.jpg", "wb") as fh:
        fh.write(base64.decodebytes(bytes(string, 'utf-8')))
    im = plt.imread("imageToSave.jpg")
    image = prepare_image(im)
    prediction = predict(image, model)
    dico = {}
    for i in prediction.keys():
        nom_latin = i
        dico[prediction[i]] = nom_latin
    return dico
