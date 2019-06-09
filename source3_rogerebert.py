"""
This module is intended to parse through movie reviews by Roger Ebert by
critics. It takes in as input an HTML page stored as text in a csv file and
aims to extract only the raw review text.
"""
import os
#import pdb
#import re
import numpy as np
import pandas as pd

#%%
os.chdir('/Users/johnpentakalos/Dropbox/UVASenior/Research/critics/')
ebert = pd.read_csv('source3 Roger Ebert.csv', encoding="ISO-8859-1")
#%%
def view_files(rev_num):
    """Used during the testing process to compare actual movie reviews with the
    cleaned text."""
    critic_stuff = open('link+review.txt', 'w')
    test_review = ebert.iloc[rev_num]['review_body_html']
    test_link = ebert.iloc[rev_num]['review_link']
    title = ebert.iloc[rev_num]['movie_title']
    critic_stuff.write(title + '\n')
    critic_stuff.write('imdb.com' + str(test_link) + '\n')
    critic_stuff.write(str(test_review))
    critic_stuff.close()
    return test_review, title

def clean_ads(text):
    """Ebert reviews are marked w Advertisement. This method takes in text w ads
    and outputs the text without any of the ads."""
    start_text = 'Advertisement'
    end_text = '});'
    start = 0
    cleaned_text = ""
    ad_start = text.find(start_text)
    #pdb.set_trace()
    while ad_start >= 0:
        cleaned_text += text[start:ad_start].strip()
        start = text.find(end_text, ad_start) + len(end_text)
        ad_start = text.find(start_text, start)
    cleaned_text += text[start:].strip()
    return cleaned_text

def find_end(text):
    """The method iterates through the list of ending texts and outputs the one
    which appears first."""
    endings = ['Photos ©', '(function() {\r\nvar zergnet', '(function() {var zergnet',\
                                         'Next Article:']
    end_indices = []
    for end in endings:
        end_indices.append(text.find(end))
    end_indices = np.array(end_indices)
    valid_indices = end_indices[end_indices > 0]
    if not valid_indices == 0:
        return -1
    return valid_indices.min()


#%%
def process_review(text):
    """Processes review as follows. Cleans out extraneous text then identifies
    start and end of review. Returns the text found in between."""
    text = text.replace('\r', '').replace('\n', '').replace('          ', '').replace('\t', '')
    text = clean_ads(text)
    start_text = 'twitter-wjs");'
    if start_text not in text:
        return 'NO TEXT'

    start = text.find(start_text) + len(start_text)
    #pdb.set_trace()
    end = find_end(text)
    text = text[start:end]
    return text

#%%
#ebert['reviews'] = ebert['review_body_html'].apply(lambda x: process_review(str(x)))
ebert['reviews'] = ebert.apply(lambda x: process_review(str(x['review_body_html'])), axis=1)
#ebert['reviews'].value_counts().head(1)
#ebert['reviews'].shape
#ebert['reviews'][1100:1200].str[-50:]
#ebert['reviews'][1100:1200]
#ebert['reviews'][1000:1100]
#%%
""" Testing Reviews
test_review, title = view_files(876)
process_review(title, test_review)
#ebert.iloc[346]['review_body_html']
#%%
a = ebert.sample(100)['reviews']
a.str[:200]
a.str[-50:]
"""

#%%
#Saving all the finished reviews
ebert.to_csv(r'/Users/johnpentakalos/Dropbox/UVASenior/Research/critics/ebertoutput.csv')
#ebert_df = pd.read_csv('ebertoutput.csv')
