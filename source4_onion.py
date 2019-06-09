"""
This module is intended to parse through movie reviews by Roger Ebert by
critics. It takes in as input an HTML page stored as text in a csv file and
aims to extract only the raw review text.
"""
import os
#import pdb
import pandas as pd

os.chdir('/Users/johnpentakalos/Dropbox/UVASenior/Research/critics/')
#%%
ONION = pd.read_csv('source4_TheOnionClub.csv', encoding="ISO-8859-1")

#%%
def view_files(rev_num, review_df):
    """Used during the testing process to compare actual movie reviews with the
    cleaned text."""
    critic_stuff = open('link+review.txt', 'w')
    test_review = review_df.iloc[rev_num]['review_body_html']
    test_link = review_df.iloc[rev_num]['review_link']
    title = review_df.iloc[rev_num]['movie_title']
    critic_stuff.write('imdb.com' + str(test_link) + '\n')
    critic_stuff.write(str(test_review))
    critic_stuff.close()
    return test_review, title
#
#test_review = view_files(1, onion)
#test_review
#%%
#def delete_iter(text, delete_phrases):
#    """The text """
#    start_text = ['Runtime', 'Cast']
#    count = 0
#    while count < len(start_text):
#        start = text.find(start_text[count])
#        if start>0:
#            text = text[start+len(start_text[count]):]
#        else:
#            count+=1
#    return text

def review_start(text):
    """Identifies the start of the review text."""
    #pdb.set_trace()
    common_doubles = ['Mc', 'Mac', 'La', '-']
    start = text[:150]
    tokenized = start.split(' ')
    #Attempts to identify the last occurence of an actors name. The idea is
    #two consecutive capitalized words signifies a name. Finding the last actor
    #full-name in the beginning section of the text was the best pattern signif-
    #ying the review had begun.
    for j in range(len(tokenized)):
        word = tokenized[j]
        capital_count = sum(1 for c in word if c.isupper())
        #Common doubles tries to exclude situations where an actors last name has
        #two capitalizations within itself.
        if capital_count >= 2 and all(substr not in word for substr in common_doubles):
            for i in range(1, len(word)):
                if word[i].isupper():
                    return text.find(word) + i
    return -1
def process_dvd_review(html_body):
    """This method is used when the associated review is for a DVD Release."""
    end_text = 'Share This Story'
    start_text = 'permalink'
    dvd_text = 'DVD Review'
    start_index = html_body.find(start_text) + len(start_text)
    end_index = html_body.find(end_text)
    html_body = html_body[start_index:end_index]
    #html_body = html_body[:html_body.rfind('Advertisement.') - len('Advertisemen')]

    html_body = html_body[html_body.rfind(dvd_text) + len(dvd_text):]

    return html_body

def process_review(title, html_body):
    """Takes in html body text and returns cleaned critic review text."""
    #Starts by removing garbage text.
    html_body = html_body.replace('\x92', '\'').replace('\r\n', ' ').replace('\x97', '')\
            .replace('Advertisement', '').replace('\x93', '').replace('\x94', '')
    end_text = 'Share This Story'
    start_text = 'permalink'
    #Returns NA for reviews that don't contain the unique end text
    if end_text not in html_body:
        return 'NA'
    start_index = html_body.find(start_text) + len(start_text)
    end_index = html_body.find(end_text)
    #pdb.set_trace()
    html_body = html_body[start_index:end_index]

    #When the cast keyword appears we look for the start text using the actor
    #name search.
    cast = html_body.rfind('Cast')
    if cast > 0:
        cast += 4
        new_start = review_start(html_body[cast:])
        if new_start > 0:
            html_body = html_body[cast+new_start:]
        else:
            html_body = html_body[cast:]

    #Trimming extra text attached in Animated movie reviews
    anim_text = 'Animated)'
    if html_body.find(anim_text) == 0:
        html_body = html_body[len(anim_text):]
    #Trimming extra text attached in DVD reviews
    if 'DVD Review' in html_body:
        title_remove = len(title)*2
        html_body = process_dvd_review(html_body)[title_remove:]
    return html_body

#%%
#Used for testing review cleaning
#test_review, title = view_files(21, onion)
#process_review(title, test_review)
#%%
ONION['reviews'] = ONION.apply(lambda x: process_review(str(x['movie_title']),\
     str(x['review_body_html'])), axis=1)
#onion['reviews']
#onion['reviews'].value_counts().head(1)

#%%
#onion_test['dvd_reviews'] = onion_test['review_body_html'][dvd_reviews].apply\
#(lambda x: process_dvd_review(str(x)))
#onion_test['dvd_reviews'] = onion_test[dvd_reviews].apply(lambda x: remove_titles\
#(str(x['dvd_reviews']), str(x['movie_title'])), axis=1)
#%%
#a = onion.sample(110)['reviews']
#a.str[:100]
#a.str[-20:]
#%%
ONION.to_csv(r'/Users/johnpentakalos/Dropbox/UVASenior/Research/critics/onionoutput.csv')
#onion_df = pd.read_csv('onionoutput.csv')
