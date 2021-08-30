from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/test-operation-bidon")
def test_operation_bidon(entered_data):
    return (f"test-operation-bidon : {int(entered_data)*2} millions de papillons")

@app.get("/predict-image")
def predict_image(image):
    return image
    # uploaded_file = np.array(uploaded_file)
    # return uploaded_file.shape

    # image = Image.open(uploaded_file)
    # img_array = np.array(image)
    # return img_array

@app.get("/predict")
def predict(uploaded_file):
    return (pd.DataFrame([[97.64,87.33,67.24,20.1,1.3],
             ['nom_latin1','nom_latin2','nom_latin3', 'nom_latin4', 'nom_latin5'],
            ['nom_commun1','nom_commun2','nom_commun3', 'nom_commun4', 'nom_commun5']]))

# joblib.load du model
# .predict sur DataFrame

# liste de résultats
# récupérer éléments zéro
