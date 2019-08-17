'''Acts as a main method which loads in data, trains a model, and outputs predictions.'''
#import sys
import multilabel_process
import model_eval
import model_load
from keras.applications import resnet50
from keras import backend as K
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
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
#%% Cell 2
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
    #Freeze all but the last specified layers.
    for layer in model.layers[:-10]:
        layer.trainable = False
    return model
#%% Cell 3
#Select which model structure we use.
MODEL = model_load.get_av_model()
#MODEL = load_resnet(True)
#Select which optimizer to use.
#MODEL.compile(loss="binary_crossentropy", optimizer='sgd', metrics=["categorical_accuracy"])
MODEL.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
print('Model loaded and compiled.')
#%% Cell 4
#Load in poster data in a usable format
DATA_PATH = '/Users/johnpentakalos/Posters/'
#DATA_PATH = sys.argv[1]
X_train, Y_train, X_validate, Y_validate, X_test, Y_test = \
    multilabel_process.img_process(DATA_PATH, 7000, 1999)
filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, \
            save_best_only=True, mode='min')
callbacks_list = [checkpoint]
print('Poster data loaded and split into train validate test.')
print('Train set has dimensions: ' + str(X_train.shape))
print('Validate set has dimensions: ' + str(X_validate.shape))
print('Test set has dimensions: ' + str(X_test.shape))

#%% Cell 5
#Model Training
history = MODEL.fit(X_train, Y_train, epochs=5, validation_data=(X_validate, Y_validate),\
          batch_size=32, callbacks=callbacks_list)
#history = MODEL.fit(X_train, Y_train, epochs=5, validation_data=(X_validate, Y_validate),\
#          batch_size=32)
MODEL.save(DATA_PATH + 'model4000_3_unfrozen.h5')
print('Model trained and saved.')
#model.fit(X_train, Y_train, epochs=2, batch_size=32)
#%% Cell 6
#Make Model Predictions
RESULTS = MODEL.predict(X_test)
print('Model predictions completed.')
#%% Cell 7
#Predictions on the training set.
#TRAIN_SET = MODEL.predict(X_train)
#print ('Model predictions on the training set.')
#train_predict, train_actual = model_eval.predict_df(Y_train, TRAIN_SET, 0)
#%% Cell 8
#Convert predictions into suitable dataframe
predictions_df, actual_df = model_eval.predict_df(Y_test, RESULTS, 0)
#Print Precision rates.
model_eval.get_precision_recall(predictions_df, actual_df)
model_eval.loss_curves(history, DATA_PATH, 'loss_4000_10', 'acc_4000_10')
