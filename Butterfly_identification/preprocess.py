import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder


def get_data (data=["train","val","test"],data_path='../raw_data/splits/'):
    '''Function to get Train, Val and Test data. That function get in input a list of
    the desired data and return a tuple containing the 3 datasets : 
    (data_train,data_val,data_test)'''

    data_train = pd.read_json(data_path + 'train.json').T
    data_val = pd.read_json(data_path + 'val.json').T
    data_test = pd.read_json(data_path + 'test.json').T

    data_dict = {"train":data_train,"val":data_val,"test":data_test}

    result = []
    for i in data :
        result.append(data_dict[i])
    return tuple(result)


def feature_engineering(df,images_path='../raw_data/IMG/'):
    '''That function add a species columns thanks to the genus and epithet columns'''
    df["path_to_image"]=images_path+df["image_name"]
    df['species'] = df['genus']+'_'+df['specific_epithet'] 
    return df


def get_data_minphoto(df, nb_min_photo_by_species = 25):
    '''This function allows to select only 25 (by default) pictures of each species (dropping those with less then 25 picures)'''
    tri_species = pd.DataFrame(df['species'].value_counts()).reset_index()
    tri_species.columns = ['species','nombre']
    tri_species['nombre'] = tri_species['nombre'].astype('uint16')
    # Save the species to keep
    keep_species = tri_species[tri_species['nombre']>=nb_min_photo_by_species]['species']
    keep_species = np.array(keep_species)
    # Create a column to filter
    df['triage'] = [ i in keep_species for i in df['species']]
    # Récupère le DF en filtrant sur les espèces de plus de 20 photos
    df_clean = df.loc[df['triage'] == True].copy()
    # Drop the column filter
    df_clean.drop('triage', axis = 1, inplace = True)
    return df_clean


def filter_val_test(df_train, df_to_filter):
    ''' This function remove from df_to_filter (df_val or df_test) all species that does not exist in train data'''
    espece = df_train.species.unique()
    def filtre(x):
        ''' This function return if a species is in the espece liste of the train data''' 
        if x not in espece:
            return False
        else :
            return True
    # Apply the function to create a column to filter
    df_to_filter['filter'] = df_to_filter['species'].apply(filtre)
    df_filtered = df_to_filter.loc[df_to_filter['filter'] == True].copy()
    # Drop the column filter
    df_filtered.drop('filter', axis = 1, inplace = True)
    df_to_filter.drop('filter', axis = 1, inplace = True)
    return df_filtered


def resampling(df):
    ''' This function return a balanced dataset from df, based on the smallest class'''
    count = pd.DataFrame(df['species'].value_counts().reset_index())
    count.columns = ['species', 'nombre']
    # Define the size of the smallest class
    minimum = min(count['nombre'])
    begin = True
    for i in count['species']:
        # Create the new dataframe with the first species
        if begin:
            new_df = df.loc[df['species'] == i].sample(minimum)
            begin = False
        # Sample for each species
        elif i not in new_df['species']:
            add_to = df.loc[df['species'] == i].sample(minimum)
            new_df = pd.concat([new_df, add_to],axis = 0)
    # Shuffle the dataset
    new_df.sample(frac=1)
    return new_df


def get_X_y(df,sampling=False,sample_size=None,resize = False,size = (448,448) ):
    '''This function return the features and target from a dataset'''
    if sampling :
        df = df.sample(sample_size, random_state = 818)
    image = []
    for i in df['path_to_image'] :
        img = Image.open(i)
        if resize :
            img=img.resize(size)
        image.append(np.array(img))
    X = np.array(image)
    y = np.array(df['species'])
    # Careful, the feature X here is not resized and the target need to be reshaped before the onehotencoder
    return X, y


def image_resizing(img,size=None):
    '''Function that resize a picture. Size is a tuple of 2 integers : (lenght,width).'''
    if size :
        img = img.resize(size)
    return img


def cat_encoder (y_train,y_val=None,y_test=None):
    '''This function return encoded targets : (y_train_cat,y_val_cat,y_test_cat).'''
    #----------- adding a second dimension to the target in order to proceed to the ohe ---------
    y_train = y_train.reshape(y_train.shape[0],1)
    y_val = y_val.reshape(y_val.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)
    #--------------------------------------- OneHotEncoding ---------------------------------
    ohe = OneHotEncoder(handle_unknown = "ignore",sparse=False)
    ohe.fit(y_train)
    result = []
    y_train_cat = ohe.transform(y_train)
    result.append(y_train_cat)
    if y_val is not None :
        y_val_cat = ohe.transform(y_val)
        result.append(y_val_cat)
    if y_test is not None:
        y_test_cat = ohe.transform(y_test)
        result.append(y_test_cat)
    return tuple(result)


if __name__=="__main__" :
    data_train,data_val,data_test = get_data (data=["train","val","test"])
    data_train=feature_engineering(data_train)
    data_val=feature_engineering(data_val)
    data_test=feature_engineering(data_test)
    X_train,y_train = get_X_y(data_train,sample_size=1000)
    X_val,y_val = get_X_y(data_val,sample_size=200)
    X_test,y_test = get_X_y(data_test,sample_size=200)
    y_train_cat,y_val_cat,y_test_cat = cat_encoder (y_train,y_val,y_test)
    print("Preprocess steps finished")
