import numpy as np
import pandas as pd
import string
import re
from collections import defaultdict
import nltk
nltk.download('stopwords')

def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

contraction_dict = {"aint": "is not", "arent": "are not","cant": "cannot", "'cause": "because", "couldve": "could have", "couldnt": "could not", "didnt": "did not",  "doesnt": "does not", "dont": "do not", "hadnt": "had not", "hasnt": "has not", "havent": "have not", "wasnt": "was not", "isnt": "is not", "shouldnt": "should not", "thats": "that is"}
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

# Misspelling corrections
mispell_dict = {"complis": "complies", "wellmade": "well-made", "onoff": "on/off", "kitchenaid": "kitchemaid", "roomba": "room", "appare": "apparel", "bodrum": "bodrum", "notly": "hotly", "kcup": "cup", "kcups": "cups", "krups": "cups", "roomthe":"room the"}
def _get_mispell(contraction_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_mispell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_text(text):
    # lower case and remove special characters\whitespaces
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    text = clean_text(text)
    text = replace_contractions(text)
    text = replace_typical_mispell(text)
    # tokenize document
    tokens = wpt.tokenize(text)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    text = ' '.join(filtered_tokens)
    return text