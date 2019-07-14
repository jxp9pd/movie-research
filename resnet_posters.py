import pandas as pd
import os
import matplotlib as mpl
from keras.applications import resnet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

if K.backend()=='tensorflow':
    K.set_image_dim_ordering("tf")
 
# Import Tensorflow with multiprocessing
mpl.rcParams['figure.figsize'] = (16.0, 8.0) 

model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape= (268, 182, 3))
model.summary()
x = model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(26, activation='sigmoid')(x)
genre_model = Model(inputs = model.input, outputs = predictions)
#genre_model.compile(loss='binary_crossentropy', optimizer='sgd')
#genre_model.compile(optimizer='sgd')
genre_model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=["accuracy"])
#Freeze all but the last layer
for layer in genre_model.layers[:-1]:
    layer.trainable = False

#%%
#
# In this section, we distribute the images across three directories, 
# one for training, one for validation and one for testing.
#

import shutil
#load in X and Y
data_path = '/Users/johnpentakalos/Posters/'
train_path = data_path + 'train/'
Y = pd.read_csv(data_path + 'genres.csv')
#%%
def prep_images(df, data_dir, data_class):
    '''splits the data into a different directory per genre.'''
    genre_list = df.columns.tolist()
    genre_list.remove('id')
    for genre in genre_list:
        path = data_dir + data_class
        new_dir = os.path.join(path, genre)
        os.mkdir(new_dir)
        file_path = path + genre + '/'
        movies_src = data_dir + 'poster_imgs/' + df.loc[df[genre] == 1]['id'] + '.jpg'
        movies_dest = file_path + df.loc[df[genre] == 1]['id'] + '.jpg'
        movies = pd.DataFrame.from_dict({'src': movies_src, 'dest':movies_dest})
        movies.apply(lambda x: shutil.copyfile(x['src'], x['dest']), axis=1)
        print(genre + ' had ' + str(len(movies)) + ' posters.')
#%%
#Only run the one time.
train_data = Y[:200]
prep_images(train_data, data_path, 'train/')
#%%
train_datagen = ImageDataGenerator(rescale=1./255)
#validation_datagen = ImageDataGenerator(rescale=1./255)
#test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_path,
    target_size=(268, 182), batch_size=16, class_mode=None)

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

#validation_generator = train_datagen.flow_from_directory(validation_dir,
#    target_size=(268, 182),
#    batch_size=10,
#    class_mode='binary')

history = genre_model.fit_generator(train_generator,
    steps_per_epoch=100,
    epochs=2)
