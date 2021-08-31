from fastapi import params
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.utils import load_img
import time
import requests
import pickle
import base64
import io
#CSS

CSS = """
.titre_principal {
    text-align: center; color : white; FONT face='century gothic';
}
"""
st.write('<style>{CSS}</style>', unsafe_allow_html=True)

#titres et textes introductifs
logo_snapillon = Image.open("/Users/prunelle/Downloads/snapillon_logo.png")
logo_snapillon  = logo_snapillon.resize((200,210))
col1, col2, col3 = st.columns(3)
with col1:
    pass
with col2:
    st.image(logo_snapillon )
with col3 :
    pass

#st.markdown("<h1 class='titre_principal'>SNAPILLON</h1>", unsafe_allow_html=True)
st.markdown("""# Par ici les petits papillons !
### Uploader une photo afin de tâcher de déterminer l'espèce de papillons associée
""")

#set background
@st.cache
def load_image(path):
    with open(path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    return encoded

def image_tag(path):
    encoded = load_image(path)
    tag = f'<img src="data:image/png;base64,{encoded}">'
    return tag

def background_image_style(path):
    encoded = load_image(path)
    style = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    '''
    return style
image_path = "/Users/prunelle/Downloads/9B2F199E-79BF-4EBD-B729-D518A58D1292_1_105_c.jpeg"
image_link = 'https://docs.python.org/3/'
st.write(background_image_style(image_path), unsafe_allow_html=True)

#drag and drop files
st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("", type=['png','jpeg','jpg'])

if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    st.write(file_details)
    image = Image.open(uploaded_file)
    st.image(image)
    st.warning('Photo chargée avec succès !')
    # utiliser request pour envoyer un fichier en method post

#barre d'avancement lors du calcul de l'algorithme
if uploaded_file is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        center_button = st.button('Lancement algorithme Snapillon')
    with col3 :
        pass
    if center_button:
        latest_iteration = st.empty()
        bar = st.progress(0)

        for i in range(100):
            # Update the progress bar with each iteration.
            latest_iteration.text(f'Iteration {i+1}')
            bar.progress(i + 1)
            time.sleep(0.04)

        st.markdown("""# Par ici les résultats !
        """)


#PICKLE CONVERTION
# st.markdown("""# Convertir au format pickle""")
# url_image = "/Users/prunelle/Downloads/test_image_pickel.pkl"
# file = open(url_image, 'wb')
# # Pickle dictionary using protocol 0.
# pickle.dump(uploaded_file, file)
# file.close()

check1= st.checkbox('launch api multiplication papillons')
#check2 = st.checkbox('launch api test image')
check3 = st.checkbox('launch api test image 2')

#retour API test connexion api : calcul multiplication papillons avec nb en entrée streamlit
if check1:
    st.markdown("""# API TEST MULTIPLICATION PAPILLONS""")
    entered_data = int(st.slider('Entrer le nombre de papillons', 1, 6, 1))
    url1 = 'http://127.0.0.1:8000/test-operation-bidon'
    parameters = dict(entered_data = entered_data)
    st.markdown(requests.get(url1, params = parameters).json())

#retour API test image en entrée renvoie 'image ok'
# if check2:
#     st.markdown("""# API TEST IMAGE""")
#     parameters2 = dict(url = url_image)
#     url2 = 'http://127.0.0.1:8000/predict-image'
#     st.markdown(requests.get(url2, params = parameters2).json())

#retour API test image en entrée renvoie dictionnaire en sortie
if check3:
    st.markdown("""# API TEST IMAGE 2""")
    #parameters3 = dict(file = uploaded_file)
    url2 = 'http://127.0.0.1:8000/predict'
    response = requests.post(
        url2, files={"file": ("media", io.BytesIO(uploaded_file.read()), "image/jpeg")}
    )
    assert response.status_code == 200
    st.warning('Photo envoyée avec succès !')
    #st.markdown(requests.post(url2, params = parameters3).json())



#retour API test model
# if uploaded_file is not None:
#     st.markdown("""# API TEST MODEL""")
#     parameters2 = dict(url = url_image)
#     url2 = 'http://127.0.0.1:8000/predict-image'
#     st.markdown(requests.get(url2, params = parameters2).json())

#affichage des résultats / retour API

# def data_transformation(df):
#     data = df.T
#     data["path_to_image"]="../raw_data/IMG/"+data["image_name"]
#     data['species'] = data['genus']+' '+data['specific_epithet']
#     return data

# df = data_transformation(pd.read_json('../raw_data/splits/train.json'))

# def find_picture(species, df, nb_image = 3):
#     data_to_sample = df.loc[df['species'] == species, 'path_to_image']
#     sampling = data_to_sample.sample(nb_image)
#     return sampling

# #retourner les trois images et le nom de l'espèce
# species = ['Parnassius apollo']
# for i in species :
#     st.markdown(f"""# Espèce identifiée : {i}""")
#     st.markdown(f"""# Nom commun : {i}""")
#     L = find_picture(i, df, nb_image = 3)
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.image(Image.open(L[0]))
#         pass
#     with col2:
#         st.image(Image.open(L[1]))
#     with col3 :
#         st.image(Image.open(L[2]))


#def jpg_image_to_array(image_path):
#   """
#   Loads JPEG image into 3D Numpy array of shape
#   (width, height, channels)
#   """
#   with Image.open(image_path) as image:
#     im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
#     im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
#   return im_arr

# if uploaded_file is not None:
#     st.write(jpg_image_to_array(uploaded_file))


#bouton "CHARGER MON IMAGE"

#image_path_2 = '/Users/prunelle/Downloads/Bouton_charger_mon_image.png'
# image_link = "https://github.com/streamlit/streamlit/issues/406"
# st.write(f'<a href="{image_link}">{image_tag(image_path_2)}</a>', unsafe_allow_html=True)

# bouton rond charger mon image
# image = Image.open('')
# image = image.resize((200,200))
# st.image(image)
# st.markdown('<div class="qqchose">{st.image(image)}</div>', unsafe_allow_html=True)
