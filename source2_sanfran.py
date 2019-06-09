"""
This module is intended to parse through movie reviews in the San Francisco
Chronicle by critics. It takes in as input an HTML page stored as text in a csv
file and aims to extract only the raw review text.
"""
import os
import re
import numpy as np
import pandas as pd
#import pdb


os.chdir('/Users/johnpentakalos/Dropbox/UVASenior/Research/critics/')
#%%
sf_df = pd.read_csv('source2_sanfran.csv', encoding="ISO-8859-1")

#%%
sf_test = sf_df[:100]

#%%
def view_files(rev_num):
    """Used during the testing process to compare actual movie reviews with the
    cleaned text."""
    critic_stuff = open('link+review.txt', 'w')
    test_review = sf_df.iloc[rev_num]['review_body_html']
    test_link = sf_df.iloc[rev_num]['review_link']
    title = sf_df.iloc[rev_num]['movie_title']
    critic_stuff.write(title + '\n')
    critic_stuff.write('imdb.com' + str(test_link) + '\n')
    critic_stuff.write(str(test_review))
    critic_stuff.close()
    return test_review, title

#%%

def find_start(text, start):
    """Aims to identify the start of a review. Each if loop corresponds to a
    different starting pattern in a movie review. They're ordered such that
    if multiple of the same start texts appear in a review, the method will
    return the last of them."""
    #pdb.set_trace()
    if start.max() > 0:
        return start.max() + text[start.max():].find(')') + 1
    if text.find('minutes.)') > 0:
        return text.find('minutes.)')  + len('minutes.)')
    if text.find('theaters.)') > 0:
        return text.find('theaters.)')  + len('theaters.)')
    if text.find('Published') > 0:
        #pdb.set_trace()
        publish_text = text.find('Published')
        if text.find("Published Date (newest first)") > 0:
            return 9999999
        return publish_text + re.search(r'[12]\d{3}', text[publish_text:]).start() + 4
    return -1

def find_rating(text):
    """Rating tends to be the very end of the meta text before the actual
    review. Method returns the start using the rating text."""
    ratings = ['R', 'PG', 'PG-13', 'G', 'NC-17', 'Not Rated', 'Not rated', 'No rating']
    start = []
    for rating in ratings:
        start.append(max(text.find('(' + rating + '.'), text.find('(Rated ' + rating),\
                                     text.find('(' + rating + ',')))
    return np.array(start)

def find_end(text):
    """Finds the end of the text. """
    endings = ['var taboola', 'var HDN', 'Advisory:']
    end_indices = []
    for end in endings:
        end_indices.append(text.find(end))
    end_indices = np.array(end_indices)
    valid_indices = end_indices[end_indices > 0]
    if not valid_indices:
        return -1
    return valid_indices.min()

def clip(text):
    """This method is used when the review closes with a gallery of images.
    It returns the earliest index of found unique gallery text."""
    endings = ['Image     1', 'Photo: ', 'Related Stories']
    end_indices = []
    for end in endings:
        end_indices.append(text.find(end))
    end_indices = np.array(end_indices)
    valid_indices = end_indices[end_indices > 0]
    if not valid_indices:
        return len(text)
    return min(valid_indices)

#%%
def clean_ads(review_text):
    """Takes in the full review text. Identifies all occurrences of unique AD
    text. Removes all ad text and returns the text sans ads."""
    START_TEXT = '/*<!'
    END_TEXT = '>*/'
    start = 0
    cleaned_text = ""
    ad_start = review_text.find(START_TEXT)
    #pdb.set_trace()
    while ad_start >= 0:
        cleaned_text += review_text[start:ad_start].strip()
        start += review_text.find(END_TEXT, start) + 1
        ad_start = review_text.find(START_TEXT, start)
    cleaned_text += review_text[start:].strip()
    return cleaned_text

def process_review(html_body):
    """Takes in the full HTML review text and returns a cleaned one by using
    the methods described above. This is used in the simpler case where the
    review url contains only the desired review. """
    #Find start
    start = find_rating(html_body)
    start_index = find_start(html_body, start)
    if start_index == 9999999:
        return 'NA'
    end_index = find_end(html_body)
    #is it a gallery?
    gallery = html_body.find('Back to Gallery')
    #Do we use back to gallery or Rating as start?
    if gallery > 0:
        tot_length = end_index - gallery
        if start_index - gallery > tot_length*0.33:
            start_index = gallery + len('Back to Gallery    ')
            end_index = start.max()

    html_body = html_body[start_index:end_index]
    html_body = html_body[:clip(html_body)]
    if html_body.startswith('.'):
        html_body = html_body[1:]
    #Remove ads
    html_body = clean_ads(html_body).replace('*/', '').strip()
    return html_body
#%%
def clean(title, text):
    """This method synthesizes the entire review cleaning process. In cases
    where the url has only one review attached it forwards to process_review.
    URLs with multiple reviews are parsed here."""
    #Remove extraneous text
    text = text.replace('\r', '').replace('\t', '').replace('\n', '')\
    .replace('                                    ', '').replace('      ', ' ')\
    .replace('mlasalle@sfchronicle.com', '')
    #pdb.set_trace()
    starts = find_rating(text)
    starts = starts[starts > 0]
    #Double check for multiple reviews
    theaters = [m.start() for m in re.finditer('theaters.\)', text)]
    #Single movie review
    if len(starts) <= 1 and len(theaters) <= 1:
        return process_review(text)

    #Cut out Gallery
    gallery = text.find('Back to Gallery') + len('Back to Gallery')
    if gallery > 0:
        text = text[gallery:]

    #Cut out all text before desired review
    review_start = text.find(title) + len(title) + 1
    if review_start < 0:
        return 'NA'
    text = text[review_start:]

    end_index = text.find('Advisory:')
    if end_index < 0:
        end_index = text.find('--')
        if end_index < 0:
            return 'NA'
    text = text[:end_index]

    starts = find_rating(text)
    starts = starts[starts > 0]
    if not starts:
        return 'NA'
    start_index = starts.max() + text[starts.max():].find(')') + 1

    text = text[:clip(text)]
    text = clean_ads(text[start_index:]).replace('*/', '').strip()
    if text.startswith('.'):
        text = text[1:].strip()
    return text

#%%
#Testing an individual review
#test_review, title = view_files(1157)
#process_review(test_review)
#
#%%
#test_review, title = view_files(5143)
#text = clean(title, test_review)
#text

#%%
#First Processing
#sf_df['reviews'] = sf_df['review_body_html'].apply(lambda x: process_review(x))
#sf_df['reviews'][:100].str[-50:]
#sf_df['reviews'][200:300]
#
#sf_df['reviews'][300:400].str[-50:]
#%%
#More advanced Processing
sf_df['reviews'] = sf_df.apply(lambda x: clean(str(x['movie_title']), \
     str(x['review_body_html'])), axis=1)
sf_test['reviews'] = sf_df.apply(lambda x: clean(str(x['movie_title']),\
       str(x['review_body_html'])), axis=1)

#sf_df[:100].reviews
#sf_df['reviews'][300:400].str[-50:]
#sf_df['reviews'].value_counts().head(2)
#%%
#TESTING
#a = sf_output.sample(110)['reviews']
#a.str[:100]
#a.str[-50:]
#Saving all the finished reviews
sf_df.to_csv(r'/Users/johnpentakalos/Dropbox/UVASenior/Research/critics/sanfranoutput.csv')
