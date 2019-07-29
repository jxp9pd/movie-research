'''Approach did not work for multi-labeling. Replaced by multilabel_process'''
import sys
import pandas as pd
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.applications import resnet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator
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
model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(268, 182, 3))
model.summary()
#Add in last layer.
x = model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(26, activation='sigmoid')(x)
genre_model = Model(inputs=model.input, outputs=predictions)

#Freeze all but the last layer
for layer in genre_model.layers[:-1]:
    layer.trainable = False
genre_model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=["accuracy", \
        "categorical_accuracy"])
#%%
data_dir = '/Users/johnpentakalos/Posters/'
#%%
#Retrain model to use 
#data_dir = sys.argv[1]
data_dir = '/Users/johnpentakalos/Posters/'
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
#test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(data_dir + 'train/', \
    target_size=(268, 182), batch_size=32, class_mode='categorical')

history = genre_model.fit_generator(train_generator, steps_per_epoch=30, epochs=5)

#for data_batch, labels_batch in train_generator:
#    print('data batch shape:', data_batch.shape)
#    print('labels batch shape:', labels_batch.shape)
#    break
#%%
test_generator = test_datagen.flow_from_directory(data_dir + 'test/', \
    target_size=(268, 182), batch_size=32, class_mode='categorical')

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
#genre_model.evaluate_generator(generator=test_generator, steps=1)
test_generator.reset()
pred=genre_model.predict_generator(test_generator, steps=STEP_SIZE_TEST+1, verbose=1)
#%%
poster_index = pd.Series(test_generator.filenames).str[:-4]
poster_index = poster_index.str.rpartition('/')[2]
genre_df = pd.read_csv(data_dir + 'genres.csv')
genre_df.set_index('id', inplace=True)
genre_list = genre_df.columns.values

predictions_df = pd.DataFrame(pred, columns=genre_list)
predictions_df.set_index(poster_index,inplace=True)
predictions_df
#%%
actual = genre_df.loc[predictions_df.index]
actual - predictions_df
#generates accuracy by genre
abs(actual-predictions_df).sum()
#Total error sum
abs(actual-predictions_df).sum().sum()
