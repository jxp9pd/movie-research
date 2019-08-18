'''Cleans training data.'''
from os import listdir
from os.path import isfile, join
import pandas as pd
pd.set_option('display.max_columns', 20)

DATA_PATH = '/Users/johnpentakalos/Posters/'
onlyfiles = [f for f in listdir(DATA_PATH + 'poster_imgs') if isfile(join(DATA_PATH\
             + 'poster_imgs', f))]
files = pd.DataFrame(onlyfiles)
files[0] = files[0].str[:-4]

#%%Cell 2
movies_csv = pd.read_csv(DATA_PATH + 'movies_list.csv', encoding = "ISO-8859-1")
movies_csv['id'] = movies_csv['movieid'].str[7:-1]

#%%Cell 3
def convert_list(genre_list):
    '''Returns genres as a python list cleaned'''
    if genre_list == None:
        return None
    return pd.Series(genre_list).str.strip().tolist()
#%%Cell 4
#Performs an inner join between posters listed in movie_list and posters downloaded.
merged = movies_csv.merge(files, left_on='id', right_on=0, how='inner')
merged_df = merged.dropna(subset=['genre'])

merged_df['genre'] = merged.genre.str.split('|')
#merged['genre'].apply(pd.Series).str.strip().tolist()
merged_df['genre_new'] = merged_df.apply(lambda x: convert_list(x['genre']), axis=1)
GENRES = merged_df['genre_new'].apply(pd.Series).stack().value_counts().index.values
GENRES = sorted(GENRES)

movies = merged_df['genre_new']
movies.index = merged_df.id

#%%Cell 5
rows = []
for item in movies.iteritems():
#    if item[0] == 'tt0298148':
#        pdb.set_trace()
    #i counts position in overall genre list
    i = 0
    #j counts position in a given movie's genre set
    j = 0
    movie_genre = sorted(item[1])
    new_row = []
    while j < len(movie_genre):
#        if GENRES[i] == 'Fantasy' and item[0] == 'tt0298148':
#            pdb.set_trace()
#        if item[0] == 'tt0298148':
#            pdb.set_trace()
        if i >= len(GENRES):
            print('Genre matching error')
            print(item)
        if movie_genre[j] == GENRES[i]:
            new_row.append(1)
            j+=1
        else:
            new_row.append(0)
        i+=1

    diff = len(GENRES) - len(new_row)
    if diff > 0:
        new_row = new_row + [0]*diff
        new_df = pd.DataFrame([new_row], columns = GENRES)
    rows.append(new_df)
#%%
movie_genres = pd.concat(rows)
movie_genres.index = merged_df.id
#%%

test = merged_df.iloc[0]
i = 0
j = 0
movie_genre = pd.Series(test).str.strip().tolist()   
new_row = []    
while j < len(movie_genre):
    if i >= len(GENRES):
        print('Genre matching error')
    if movie_genre[j] == GENRES[i]:
        new_row.append(1)
        j+=1
    else:
        new_row.append(0)
    i+=1

diff = len(GENRES) - len(new_row)
if diff > 0:
    new_row = new_row + [0]*diff

print(new_row)
print(len(new_row))


