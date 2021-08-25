import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, Sequential


def image_resizing(img,size=None):
    '''Function that resize a picture. Size is a tuple of 2 integers : (lenght,width).'''
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

def feature_engineering(df):
    '''That function add a species columns thanks to the genus and epithet columns'''
    df["path_to_image"]="../raw_data/IMG/"+df["image_name"]
    df['species'] = df['genus']+'_'+df['specific_epithet'] 
    return df

def set_nontrainable_layers(imported_model):
    '''set imported model layers' as non trainable'''
    imported_model.trainable = False
    return imported_model

def get_X_y(df,sample_size):
    '''This function return the features and target from a dataset'''
    data_sample = df.sample(sample_size, random_state = 818)
    image = []
    for i in data_sample['path_to_image'] :
        img = Image.open(i)
        #img = image_resizing(img)
        image.append(np.array(img))
    X = np.array(image)
    y = np.array(data_sample['species'])
    #careful, the feature X here is not resized and the target need to be reshaped before the onehotencoder
    return X, y

def get_data (data=["train","val","test"]):
    '''Function to get Train, Val and Test data. That function get in input a list of
    the desired data and return a tuple containing the 3 datasets : 
    (data_train,data_val,data_test)'''

    data_train = pd.read_json('../raw_data/splits/train.json').T
    data_val = pd.read_json('../raw_data/splits/val.json').T
    data_test = pd.read_json('../raw_data/splits/test.json').T

    data_dict = {"train":data_train,"val":data_val,"test":data_test}

    result = []
    for i in data :
        result.append(data_dict[i])
    return tuple(result)

def get_updated_model(imported_model,target):
    '''Take a pre-trained model, set its parameters as non-trainables, and add additional 
    trainable layers'''
    
    n_classes = pd.Series(target).nunique()
    imported_model = set_nontrainable_layers(imported_model)

    flattening_layer = layers.Flatten()
    dense_layer = layers.Dense(130, activation='relu')
    prediction_layer = layers.Dense(n_classes, activation='softmax')

    model = Sequential([imported_model,flattening_layer,dense_layer,prediction_layer])
    return model

def model_compile(model,learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
    adam = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=False,
    name='Adam', **kwargs)
    model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
    return model

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
