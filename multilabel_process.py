'''Loads in images, pre-processes them for keras, splits data into train test validate'''
from keras.preprocessing import image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#%%
def convert_image(img_url):
    '''Converts a single image into a numpy matrix'''
    img = image.load_img(img_url, target_size=(268, 182, 3))
    img = image.img_to_array(img)
    img = img/255
    return img
#%%
def process_imgs(movie_df, data_dir):
    '''Loads jpg and converts to an appropriate img array. Takes in the directory
    of all the images + movie list and outputs a saveable float array dataframe.'''
    #import pdb
    #pdb.set_trace()
    image_list = data_dir + 'poster_imgs/' + movie_df['id'] + '.jpg'
    imgs_processed = []
    for index, img_url in image_list.iteritems():
        imgs_processed.append(convert_image(img_url))
    #imgs_processed = image_list.apply(lambda x: convert_image(x))
    return np.array(imgs_processed)

#%%
def img_process(data_path, size):
    '''Splits the data up into Train, Test, Validate. Then makes the succesive
    calls to process_imgs'''
    #data_path = '/Users/johnpentakalos/Posters/'
    genres = pd.read_csv(data_path + 'genres.csv')[:size]
    train_data, double_data = train_test_split(genres, test_size=0.2, random_state=1)
    validate_data, test_data = train_test_split(double_data, test_size=0.5, random_state=1)
   # pdb.set_trace()
    x_train = process_imgs(train_data, data_path)
    y_train = np.array(train_data.drop('id', axis=1))
    x_validate = process_imgs(validate_data, data_path)
    y_validate = np.array(validate_data.drop('id', axis=1))
    x_test = process_imgs(test_data, data_path)
    y_test = np.array(test_data.drop('id', axis=1))

    return x_train, y_train, x_validate, y_validate, x_test, y_test
