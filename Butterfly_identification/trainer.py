import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Sequential
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
import Butterfly_identification.preprocessbutterfly as preproc
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.strings import split
from tensorflow.io import decode_jpeg, read_file
from tensorflow import argmax
import matplotlib.pyplot as plt
import os

AUTOTUNE = AUTOTUNE
def get_generators(df_train,df_val,df_test, VGG16 = True):
    '''Use filtered dataframe to create dataframe generators train_ds,val_ds and test_ds'''
    
    train_names = df_train['species'].drop_duplicates()

    def get_label(file_path,class_names=train_names):
        # convert the path to a list of path components
        parts = split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == class_names
        # Integer encode the label
        return argmax(one_hot)

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = decode_jpeg(img, channels=3)
        if VGG16 ==True :
            img = preprocess_input(img)
        # resize the image to the desired size
        return img
    
    def process_path(file_path, class_names=train_names):
        label = get_label(file_path, class_names)
        # load the raw data from the file as a string
        img = read_file(file_path)
        img = decode_img(img)
        return img, label

    def configure_for_performance(ds):
        #ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size=32)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    parts =  os.getcwd().split("/")
    ABS_PATH ="/"+parts[1]+"/"+parts[2]
    PATH_TRAIN = ABS_PATH + '/code/Em3line/Butterfly_identification/raw_data/IGM_labels/Train/'
    PATH_VAL = ABS_PATH + '/code/Em3line/Butterfly_identification/raw_data/IGM_labels/Val/'
    PATH_TEST = ABS_PATH + '/code/Em3line/Butterfly_identification/raw_data/IGM_labels/Test/'
    print(PATH_TEST)

    black_list = list(PATH_TRAIN + df_train['image_path'])
    black_list_val = list(PATH_VAL + df_val['image_path'])
    black_list_test = list(PATH_TEST + df_test['image_path'])
    print(11*"-",black_list[0])

    filtered_train = [species for species in black_list]
    filtered_val = [species for species in black_list_val]
    filtered_test = [species for species in black_list_test]
    print(11*"-",filtered_train[0])

    train_ds = Dataset.from_tensor_slices(filtered_train)
    val_ds = Dataset.from_tensor_slices(filtered_val)
    test_ds = Dataset.from_tensor_slices(filtered_test)

    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    test_ds = configure_for_performance(test_ds)

    return train_ds,val_ds,test_ds


def set_nontrainable_layers(imported_model):
    '''set imported model layers' as non trainable'''
    imported_model.trainable = False
    return imported_model

def model_compile(model,learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
    '''compile your model with hyperparameters : learning-rate,beta_A,beta_2,epsilon.'''
    adam = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=False,
    name='Adam')
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
    return model

def get_updated_ResNet(train_ds, val_ds, test_ds, IMG_SIZE,  patience=2, learning_rate=0.001, nb_epochs=15, 
                      nb_couches_dense_layer=130,Aug = False,rot = 0.2):
    imported_model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (IMG_SIZE,IMG_SIZE,3))

    #n_classes = pd.Series(y_train).nunique()
    
    # Set the first layers : 
    data_augmentation = Sequential([
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                layers.experimental.preprocessing.RandomRotation(rot),
                ])

    resize_and_rescale = Sequential([
                layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
                layers.experimental.preprocessing.Rescaling(1./255)
                ])

    # Set the first layers of the downloaded model to be untrainable
    imported_model = set_nontrainable_layers(imported_model)
    #add last layers
    flattening_layer = layers.Flatten()
    dense_layer = layers.Dense(nb_couches_dense_layer, activation='relu')
    prediction_layer = layers.Dense(n_classes, activation='softmax')
    
    if Aug == True :
        model = Sequential([data_augmentation,resize_and_rescale,imported_model,flattening_layer,dense_layer,prediction_layer])
    else :
        model = Sequential([resize_and_rescale,imported_model,flattening_layer,dense_layer,prediction_layer])
    #build model
    model = model_compile(model,learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    
    #set earlystopping
    es = EarlyStopping(monitor='val_loss', mode='max', patience=patience, verbose=1, restore_best_weights=True)
    
    #fit model
    history = model.fit(train_ds, 
                    validation_data=(val_ds), 
                    epochs=nb_epochs,
                    callbacks=[es])
    
    #evaluate model
    res_vgg = model.evaluate(test_ds)
    
    test_accuracy_vgg = res_vgg[-1]
    
    return (f"test_accuracy = {round(test_accuracy_vgg,2)*100} %"), history


def get_updated_VGG16(train_ds, val_ds, test_ds, IMG_SIZE=128,  patience=2, learning_rate=0.001, nb_epochs=15, 
                      nb_couches_dense_layer=130,Aug = False,rot = 0.2):
    '''Take a pre-trained model : ("VGG16" or "ResNet"), set its parameters as non-trainables, and add additional 
    trainable layers with a free number of neurons before compiling and fitting'''
    imported_model = VGG16(weights="imagenet", include_top=False, input_shape = (IMG_SIZE,IMG_SIZE,3))
        
        
    #n_classes = pd.Series(y_train).nunique()
    
    # Set the first layers : 
    data_augmentation = Sequential([
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                layers.experimental.preprocessing.RandomRotation(rot),
                ])

    resize_and_rescale = Sequential([
                layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
                layers.experimental.preprocessing.Rescaling(1./255)
                ])

    # Set the first layers of the downloaded model to be untrainable
    imported_model = set_nontrainable_layers(imported_model)
    #add last layers
    flattening_layer = layers.Flatten()
    dense_layer = layers.Dense(nb_couches_dense_layer, activation='relu')
    prediction_layer = layers.Dense(228, activation='softmax')
    
    if Aug == True :
        model = Sequential([data_augmentation,resize_and_rescale,imported_model,flattening_layer,dense_layer,prediction_layer])
    else :
        model = Sequential([resize_and_rescale,imported_model,flattening_layer,dense_layer,prediction_layer])
    
    #build model
    model = model_compile(model,learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    
    #set earlystopping
    es = EarlyStopping(monitor='val_loss', mode='max', patience=patience, verbose=1, restore_best_weights=True)
    
    #fit model
    history = model.fit(train_ds, 
                    validation_data=(val_ds), 
                    epochs=nb_epochs,
                    callbacks=[es])
    
    #evaluate model
    res_vgg = model.evaluate(test_ds)
    
    test_accuracy_vgg = res_vgg[-1]
    
    return (f"test_accuracy = {round(test_accuracy_vgg,2)*100} %"), history

def plot_history(history, title='', axs=None, exp_name=""):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label='train' + exp_name)
    ax1.plot(history.history['val_loss'], label='val' + exp_name)
    #ax1.set_ylim(0., 2.2)
    ax1.set_title('loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
    #ax2.set_ylim(0.25, 1.)
    ax2.set_title('Accuracy')
    ax2.legend()
    return (ax1, ax2)

# import progressbar
# from time import sleep
# def bar_progress ():
#     bar = progressbar.ProgressBar(maxval=20, \
#     widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
#     bar.start()
#     for i in xrange(20):
#         bar.update(i+1)
#         sleep(0.1)
#     bar.finish()



if __name__=="__main__" :
    df_train,df_val,df_test = preproc.get_data ()
    df_train,df_val,df_test=preproc.feature_engineering(df_train,df_val,df_test)
    df_train = preproc.get_data_minphoto(df_train, nb_min_photo_by_species = 25)
    df_val = preproc.filter_val_test(df_train,df_val)
    df_test = preproc.filter_val_test(df_train,df_test)
    print(" \n>>>>>>>>>>>>>>>     Preprocess steps finished     <<<<<<<<<<<<<<<\n \n ")
    train_ds,val_ds,test_ds = get_generators(df_train, df_val, df_test)
    print(" \n>>>>>>>>>>>>>>>        Generators created         <<<<<<<<<<<<<<<\n \n ")
    accuracy,history = get_updated_VGG16(train_ds, val_ds, test_ds, IMG_SIZE=128,  patience=2, learning_rate=0.001, nb_epochs=1, 
                      nb_couches_dense_layer=130,Aug = False,rot = 0.2)
    print(" \n>>>>>>>>>>>>>>>      Model built and trained      <<<<<<<<<<<<<<<\n \n ")
    print(accuracy)
    plot_history(history)
    print(" \n>>>>>>>>>>>>>>>            Predictions            <<<<<<<<<<<<<<<\n \n ")

