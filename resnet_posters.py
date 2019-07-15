import sys
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.applications import resnet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
#ssh jxp9pd@gpusrv04.cs.virginia.edu
#'/af12/jxp9pd/Posters/train/'
#''/Users/johnpentakalos/Posters/train/''
#source ~/pythonenv/bin/activate
#export PATH=$HOME/bin:$PATH
# Import Tensorflow with multiprocessing
if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("tf")
#mpl.rcParams['figure.figsize'] = (16.0, 8.0) 
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
genre_model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=["accuracy"])
#%%
train_datagen = ImageDataGenerator(rescale=1./255)
#validation_datagen = ImageDataGenerator(rescale=1./255)
#test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(sys.argv[1],
    target_size=(268, 182), batch_size=32, class_mode='categorical')


history = genre_model.fit_generator(train_generator, steps_per_epoch=100, epochs=2)

#for data_batch, labels_batch in train_generator:
#    print('data batch shape:', data_batch.shape)
#    print('labels batch shape:', labels_batch.shape)
#    break
#validation_generator = train_datagen.flow_from_directory(validation_dir,
#    target_size=(268, 182),
#    batch_size=10,
#    class_mode='binary')
