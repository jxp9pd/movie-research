'''This module handles the entire model training process.'''
import sys
import multilabel_process
from keras.applications import resnet50
from keras import backend as K
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
import pandas as pd
import matplotlib.pyplot as plt

#ssh jxp9pd@gpusrv04.cs.virginia.edu
#'/af12/jxp9pd/Posters/train/'
#'/Users/johnpentakalos/Posters/train/'
#source ~/pythonenv/bin/activate
#export PATH=$HOME/bin:$PATH
# Import Tensorflow with multiprocessing
if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("tf")
#mpl.rcParams['figure.figsize'] = (16.0, 8.0)
pd.set_option('display.max_columns', 26)
#%%
def load_resnet(finetune):
    '''Load in the pre-trained Resnet50 model'''
    resnet_model = resnet50.ResNet50(weights='imagenet', include_top=False,\
                                     input_shape=(268, 182, 3))
    resnet_model.summary()
    if not finetune:
        return resnet_model
    #Add in last layer.
    link = resnet_model.output
    link = GlobalAveragePooling2D()(link)
    predictions = Dense(26, activation='sigmoid')(link)
    model = Model(inputs=resnet_model.input, outputs=predictions)
    #Freeze all but the last layer
    for layer in model.layers[:-1]:
        layer.trainable = False
    return model
#%%
model = load_resnet(True)
model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=["categorical_accuracy"])
#%%
#Load in poster data in a usable format
#data_path = sys.argv[1]
DATA_PATH = '/Users/johnpentakalos/Posters/'
X_train, Y_train, X_validate, Y_validate, X_test, Y_test = \
    multilabel_process.img_process(DATA_PATH, 3000)
#%%
#Model Training
history = model.fit(X_train, Y_train, epochs=3, validation_data=(X_validate, Y_validate),\
          batch_size=32)
model.save('model2400_3.h5') 
#model.fit(X_train, Y_train, epochs=2, batch_size=32)
#%%
#Model Predictions
