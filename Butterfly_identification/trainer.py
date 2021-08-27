import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Sequential
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping

def set_nontrainable_layers(imported_model):
    '''set imported model layers' as non trainable'''
    imported_model.trainable = False
    return imported_model

def model_compile(model,learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
    '''compile your model with hyperparameters : learning-rate,beta_A,beta_2,epsilon.'''
    adam = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=False,
    name='Adam')
    model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
    return model

def get_updated_model(model,X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat,y_train, 
                      image_size,  patience=2, learning_rate=0.001, nb_epochs=15, 
                      nb_couches_dense_layer=130):
    '''Take a pre-trained model : ("VGG16" or "ResNet"), set its parameters as non-trainables, and add additional 
    trainable layers with a free number of neurons before compiling and fitting'''
    if model =='VGG16':
        imported_model = VGG16(weights="imagenet", include_top=False, input_shape = image_size)
    elif model =='ResNet':
        imported_model = ResNet50(weights = 'imagenet', 
                                             include_top = False, input_shape = image_size)
        
    n_classes = pd.Series(y_train).nunique()
    
    # Set the first layers to be untrainable
    imported_model = set_nontrainable_layers(imported_model)
    #add last layers
    flattening_layer = layers.Flatten()
    dense_layer = layers.Dense(nb_couches_dense_layer, activation='relu')
    prediction_layer = layers.Dense(n_classes, activation='softmax')
    
    updated_model = Sequential([imported_model,flattening_layer,dense_layer,prediction_layer])
    
    #build model
    updated_model = model_compile(updated_model,learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    
    #set earlystopping
    es = EarlyStopping(monitor='val_loss', mode='max', patience=patience, verbose=1, restore_best_weights=True)
    
    #fit model
    history = updated_model.fit(X_train, y_train_cat, 
                    validation_data=(X_val, y_val_cat), 
                    epochs=nb_epochs, 
                    batch_size=16, 
                    callbacks=[es])
    
    #evaluate model
    res_vgg = updated_model.evaluate(X_test, y_test_cat)
    
    test_accuracy_vgg = res_vgg[-1]
    
    return (f"test_accuracy({model} = {round(test_accuracy_vgg,2)*100} %"), history

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