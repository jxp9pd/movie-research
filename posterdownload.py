"""
Downloads posters for all the movies listed in movieposter.csv
"""
#John Pentakalos
#06-07-2019
import os
import urllib.request as req
from bs4 import BeautifulSoup
import requests
import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)

#Sets the current working directory to the one holding movieposter.csv
os.chdir('C:/Users/johnp/Dropbox/UVASenior/Research/posters/')
#os.chdir('D:/poster_imgs')

#%%
def get_url(movie_url):
    """Takes in the IMDB movie url and returns the URL for the poster image"""
    page = requests.get(movie_url)
    #Checks to see if the movie page is up.
    if page.status_code != 200:
        return None
    #IMDB pages contain a div labeled class poster containing a single poster image
    soup = BeautifulSoup(requests.get(movie_url).content.decode("utf-8"))
    div = soup.find('div', {'class':'poster'})
    #Error check if the IMDB page doesn't contain a poster div
    if div is None:
        print(movie_url + ' has no poster found')
        return None
    #Returns the img link associated with the poster image
    return div.find('img')['src']



def get_poster(url, movie_url):
    """Extract poster from a single page. Takes in IMDB URL and poster img URL."""
    #poster_id refers to the unique IMDB ID assigned to each movie
    poster_id = movie_url[26:-1]
    #directory = 'poster_imgs/' + str(poster_id) + '.jpg'
    #End location in which the posters are stored.
    directory = 'D:/poster_imgs/5000/' + str(poster_id) + '.jpg'
    if url is None:
        print(movie_url + ' is invalid')
    else:
        req.urlretrieve(url, directory)

#%%
POSTERS_FULL = pd.read_csv('movieposter.csv', encoding="ISO-8859-1")
POSTERS = POSTERS_FULL[4000:5000]
#%%
#Get all the valid links for the posters
POSTERS_LINKS = POSTERS.apply(lambda x: get_url(x['url']), axis=1)
POSTERS['poster_link'] = POSTERS_LINKS

#Get all the actual posters
POSTERS.apply(lambda x: get_poster(x['poster_link'], x['url']), axis=1)
