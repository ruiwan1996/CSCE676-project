import pandas as pd
import re
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


def get_bag_of_words(df=pd.read_csv('mbti_1.csv'), word_column='posts', min_df=2):

    posts_by_user = [row[word_column].split('|||') for _, row in df.iterrows()]
    corpus = [' '.join([preprocess_post(post) for post in posts]) for posts in posts_by_user]

    vectorizer = CountVectorizer(min_df=min_df)
    X = vectorizer.fit_transform(corpus)

    return vectorizer.get_feature_names(), X