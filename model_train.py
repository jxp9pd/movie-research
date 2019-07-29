'''This module handles the entire model training process.'''
import sys
import multilabel_process
from keras.applications import resnet50
from keras import backend as K
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
import pandas as pd

#ssh jxp9pd@gpusrv04.cs.virginia.edu
#'/af12/jxp9pd/Posters/'
#'/Users/johnpentakalos/Posters/'
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
MODEL = load_resnet(True)
#MODEL.compile(loss="binary_crossentropy", optimizer='sgd', metrics=["categorical_accuracy"])
MODEL.compile(loss="binary_crossentropy", optimizer='adam', metrics=["categorical_accuracy"])
print('Pretrained Resnet model loaded and compiled.')
#%%
#Load in poster data in a usable format
#data_path = sys.argv[1]
DATA_PATH = '/Users/johnpentakalos/Posters/'
#DATA_PATH = sys.argv[1]
X_train, Y_train, X_validate, Y_validate, X_test, Y_test = \
    multilabel_process.img_process(DATA_PATH, 5000)
print('Poster data loaded and split into train validate test.')
print('Train set has dimensions: ' + str(X_train.shape))
print('Validate set has dimensions: ' + str(X_validate.shape))
print('Test set has dimensions: ' + str(X_test.shape))
#%%
#Model Training
history = MODEL.fit(X_train, Y_train, epochs=2, validation_data=(X_validate, Y_validate),\
          batch_size=32)
MODEL.save(DATA_PATH + 'model4000_3.h5')
print('Model trained and saved.')
#model.fit(X_train, Y_train, epochs=2, batch_size=32)
#%%
#Make Model Predictions
RESULTS = MODEL.predict(X_test)
#%%
#Convert predictions into suitable dataframe
genre_df = pd.read_csv(DATA_PATH + 'genres.csv')
genre_df.set_index('id', inplace=True)
genre_list = genre_df.columns.values
actual_df = pd.DataFrame(Y_test, columns=genre_list)
predictions_df = pd.DataFrame(RESULTS, columns=genre_list)
predictions_df[predictions_df < 0.5] = 0
predictions_df[predictions_df > 0.5] = 1
predictions_df.head()
actual_df.head()
