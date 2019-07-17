import sys
import pandas as pd
import numpy as np
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.applications import resnet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
import multilabel_process

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
'''Load in the pre-trained Resnet50 model'''
resnet_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(268, 182, 3))
resnet_model.summary()
#Add in last layer.
x = resnet_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(26, activation='sigmoid')(x)
model = Model(inputs=resnet_model.input, outputs=predictions)

#Freeze all but the last layer
for layer in model.layers[:-1]:
    layer.trainable = False
model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=["accuracy", \
        "categorical_accuracy"])
#%%
'''Load in poster data in a usable format'''
#data_path = sys.argv[1]
data_path = '/Users/johnpentakalos/Posters/'
X_train, Y_train, X_validate, Y_validate, X_test, Y_test = \
    multilabel_process.img_process(data_path, 3000)
#%%
'''Model training'''
model.fit(X_train, Y_train, steps_per_epoch=30, epochs=2, validation_data=\
          (X_validate, Y_validate), batch_size=32)
#model.fit(X_train, Y_train, epochs=2, batch_size=32)
#%%
'''Model Predictions'''
