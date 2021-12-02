from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import pandas as pd
import pickle
import os
from tensorflow.keras.applications.imagenet_utils import (decode_predictions,
                                                          preprocess_input)
PATH = 'raw_data/'
def load_model():
    # loads and returns teh pretrained model # CHANGE PATH
    filepath = PATH + "Docker/models/API_FTW"
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
    infile = open(PATH + "Docker/species_table",'rb')
    table = pickle.load(infile)
    infile.close()
    # Merge results and total
    table = pd.DataFrame(table).reset_index(drop = True)
    total = resultat.merge(table, right_index=True, left_index=True)
    total.columns = ['score', 'species']
    #= Créer la table de correspondance avec les noms communs
    table_nom_commun = pd.read_json(PATH + 'Docker/table_commun.json')
    table_nom_commun = table_nom_commun.replace(' ', '_', regex = True)
    table_nom_commun.columns = ['species', 'nom_commun']

    # Affiche la table de prédiction
    table_pred = total.merge(table_nom_commun, how = 'left', on = 'species')
    total_sort = table_pred.sort_values(ascending=False, by = 'score').head(5)

    # Sort results and keep the 5 better = CHECK THE NUMBER TO KEEP
    total_sort = total_sort.set_index('species', drop = True).to_dict()['score']
    print(f'tableau prédiction avec nom commun : {total_sort}')
    return total_sort


#affichage des résultats / retour API
# CHANGE DEFAULT PATH

def get_prediction_pictures(species, path = PATH):
    ''' This function take one species, the path of the photos folder and the number of photos to return and create a pickle file with the number of photos by species and return the path of this pickle file'''
    new_path = path + 'IGM_labels/Train/' + species
    all_photos = os.listdir(new_path)
    file = open(path + 'Pickle/' + f'pickle_{species}.pkl','wb')
    dico = {}
    for num, name_path in enumerate(all_photos) :
        img = Image.open(new_path + '/' + name_path)
        dico[num] = img
    pickle.dump(dico,file)
    file.close()
    return path + 'Pickle/' + f'pickle_{species}.pkl'
