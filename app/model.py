from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import pandas as pd
import pickle
import os
from tensorflow.keras.applications.imagenet_utils import (decode_predictions,
                                                          preprocess_input)

def load_model():
    # loads and returns teh pretrained model # CHANGE PATH
    filepath = "/Users/prunelle/code/Em3line/Butterfly_identification/raw_data/Docker/models/API-FTW"
    model = load_keras_model(filepath)
    # compile=True, options=None)
    return model

# CHARGER IMAGE PICKLE OU BYTES

def prepare_image (img):
    ''' Preprocess the image before model prediction'''
#     img = img.resize(target)
#     img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img,axis = 0)
    return img


def predict(image, model):
    #Prediction results
    resultat = pd.DataFrame(model.predict(image)).T

    # Load prediction table of species
    infile = open("raw_data/Docker/species_table",'rb')
    table = pickle.load(infile)
    infile.close()

    # Merge resultat and total
    table = pd.DataFrame(table).reset_index(drop = True)
    total = resultat.merge(table, right_index=True, left_index=True)
    total.columns = ['score', 'species']

    # Sort results and keep the 5 better = CHECK THE NUMBER TO KEEP
    total_sort = total.sort_values(ascending=False, by = 'score').head(5)
    total_sort = total_sort.set_index('species', drop = True).to_dict()['score']
    return total_sort


#affichage des résultats / retour API
# CHANGE DEFAULT PATH

def get_prediction_pictures(species, path = 'raw_data/Docker/'):
    ''' This function take one species, the path of the photos folder and the number of photos to return and create a pickle file with the number of photos by species and return the path of this pickle file'''
    new_path = path + 'Photos/' + species
    all_photos = os.listdir(new_path)
    file = open(path + 'Pickle/' + f'pickle_{species}.pkl','wb')
    dico = {}
    for num, name_path in enumerate(all_photos) :
        img = Image.open(new_path + '/' + name_path)
        dico[num] = img
    pickle.dump(dico,file)
    file.close()
    return path + 'Pickle/' + f'pickle_{species}.pkl'
