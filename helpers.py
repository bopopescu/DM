# from spacy.lang.en import English
from spacy.lang.ro import Romanian
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

import nltk
nltk.download('wordnet')

# parser = English()
parser = Romanian()

nltk.download('stopwords')

# en_stop = set(nltk.corpus.stopwords.words('english'))
en_stop = set(nltk.corpus.stopwords.words('romanian'))

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

def prepare_text_for_lda_2(text): # without lemma
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 3]
    tokens = [token for token in tokens if token not in en_stop]
    return tokens