import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

def filter_word(word):
    if bool(re.search(r'[\d]', word)) or \
            bool(re.search('[^\w]', word)) or \
            bool(re.search(r'http', word)) or \
            word in stop_words:
        return False
    else:
        return True

def preprocess_post(post):
    post = word_tokenize(post)
    # remove stop words + stem
    post = [porter.stem(word.lower()) for word in post if filter_word(word.lower())]
    post = ' '.join(post)
    return post


def get_ngrams(n, word_column='posts', df=pd.read_csv('mbti_1.csv')):

    posts_by_user = [row[word_column].split('|||') for _, row in df.iterrows()]
    corpus = [' '.join([preprocess_post(post) for post in posts]) for posts in posts_by_user]

    ngram_vectorizer = CountVectorizer(binary=False, ngram_range=(n, n), min_df=2)
    X = ngram_vectorizer.fit_transform(corpus)

    return ngram_vectorizer.get_feature_names(), X