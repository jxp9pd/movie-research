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
        movie_copy = set()
        if os.path.isdir(new_dir):
            #If folder already exists, only include new posters
            print('Folder ' + genre + ' already exists') 
            a = df.loc[df[genre] == 1]['id'] + '.jpg'
            b = pd.Series(os.listdir(new_dir))
            full = pd.concat([a,b])
            movie_copy = full.drop_duplicates(keep=False)
            print ('Posters existing: ' + str(len(b)))
            print ('Posters added: ' + str(len(movie_copy)))
        else:
            os.mkdir(new_dir)
            movie_copy = df.loc[df[genre] == 1]['id'] + '.jpg'
        file_path = path + genre + '/'
        movies_src = data_dir + 'poster_imgs/' + movie_copy
        movies_dest = file_path + movie_copy
        movies = pd.DataFrame.from_dict({'src': movies_src, 'dest':movies_dest})
        movies.apply(lambda x: shutil.copyfile(x['src'], x['dest']), axis=1)
        print(genre + ' has ' + str(len(os.listdir(new_dir))) + ' posters.')

#%%
if __name__ == ("__main__"):
    print(sys.argv[1])
    data_path = sys.argv[1]
    train_path = data_path + 'train/'
    Y = pd.read_csv(data_path + 'genres.csv')
    test_data = Y[:1000]
    prep_images(test_data, data_path, 'train/')
#%%

