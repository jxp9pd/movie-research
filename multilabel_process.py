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
def date_split(data_path, first_date, movies):
    '''Trims to only include posters produced after the input date'''
    full_df = pd.read_csv(data_path + 'movies_list.csv')
    #Takes out just the unique IDs from the IMDB urls
    full_df.index = full_df['movieurl'].str[26:-1]
    full_df = full_df.loc[movies]
    full_df['release_year'] = full_df.year.str[1:5]
    #Trim non-dated posters
    full_df = full_df[full_df['release_year'] != "????"]
    full_df['release_year'] = full_df['release_year'].astype('int32')
    full_df = full_df[full_df['release_year'] >= first_date]
    #just need to return the list of movies.
    return full_df.index
    

#%%
def img_process(data_path, size, year):
    '''Splits the data up into Train, Test, Validate. Then makes the succesive
    calls to process_imgs'''
    genres = pd.read_csv(data_path + 'genres.csv')[:size]
    genres.index = genres['id']
    valid_index = date_split(data_path, year, genres.index)
    genres = genres.loc[valid_index].dropna()
    #randomizes the order of the dataset
    genres = genres.sample(frac=1)
    train_data, double_data = train_test_split(genres, test_size=0.2, random_state=1)
    validate_data, test_data = train_test_split(double_data, test_size=0.5, random_state=1)
    x_train = process_imgs(train_data, data_path)
    y_train = np.array(train_data.drop('id', axis=1))
    x_validate = process_imgs(validate_data, data_path)
    y_validate = np.array(validate_data.drop('id', axis=1))
    x_test = process_imgs(test_data, data_path)
    y_test = test_data.set_index('id')

    return x_train, y_train, x_validate, y_validate, x_test, y_test

