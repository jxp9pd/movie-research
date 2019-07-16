from keras.preprocessing import image
import pandas as pd
from sklearn.model_selection import train_test_split
#%%
def convert_image(img_url):
    '''Converts a single image into a numpy matrix'''
    img = image.load_img(img_url)
    img = image.img_to_array(img)
    img = img/255
    return img
#%%
def process_imgs(movie_df, data_dir):
    '''Loads jpg and converts to an appropriate img array. Takes in the directory
    of all the images + movie list and outputs a saveable float array dataframe.'''
    image_list = data_dir + 'poster_imgs/' + movie_df['id'] + '.jpg'
    imgs_processed = image_list.apply(lambda x: convert_image(x))
    return imgs_processed
    #length = len(imgs_processed)
    #np.save(data'img_data' + str(length) + '.npy', imgs_processed)
    

#%%
def img_process(data_path, size):
    #data_path = '/Users/johnpentakalos/Posters/'
    Y = pd.read_csv(data_path + 'genres.csv')
    if size != len(Y):
        Y = Y[:size]
    train_data, double_data = train_test_split(Y, test_size=0.2, random_state=1)
    validate_data, test_data = train_test_split(double_data, test_size=0.5, random_state=1)
    return process_imgs(train_data, data_path), process_imgs(validate_data, data_path),\
        process_imgs(test_data, data_path)
    
