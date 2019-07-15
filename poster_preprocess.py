import os
import shutil
import sys
import pandas as pd
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
data_path = sys.argv[0]
train_path = data_path + 'train/'
Y = pd.read_csv(data_path + 'genres.csv')
#%%
test_data = Y[:100]
