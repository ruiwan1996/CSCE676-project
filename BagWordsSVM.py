import pandas as pd
from get_bag_of_words import preprocess_post, get_bag_of_words
from sklearn.svm import SVC
import collections

class SVMWithBagOfWords:
    def __init__(self):
        '''
        mbti_type_series = df['type']
        '''
        words, bag_of_words = get_bag_of_words()
        df = pd.read_csv('mbti_1.csv')
        mbti_type_series = df['type']

        self.words = words
        self.i_classifier = None
        self.n_classifier = None
        self.t_classifier = None
        self.j_classifier = None
        self.__train__(mbti_type_series, bag_of_words)

    def __train__(self, mbti_type_series, bag_of_words):
        y = mbti_type_series

        self.i_classifier = SVC()
        self.i_classifier.fit(bag_of_words, y.apply(lambda x: x[0] == 'I'))

        self.n_classifier = SVC()
        self.n_classifier.fit(bag_of_words, y.apply(lambda x: x[1] == 'N'))

        self.t_classifier = SVC()
        self.t_classifier.fit(bag_of_words, y.apply(lambda x: x[2] == 'T'))

        self.j_classifier = SVC()
        self.j_classifier.fit(bag_of_words, y.apply(lambda x: x[3] == 'J'))

    def classify(self, post):
        corpus = preprocess_post(post)
        counter = collections.Counter(corpus.split())
        bow_row = [[counter.get(word, 0) for word in self.words]]

        answer = [None, None, None, None]

        i = self.i_classifier.predict(bow_row)
        if i[0] == 1:
            answer[0] = 'I'
        else:
            answer[0] = 'E'

        n = self.n_classifier.predict(bow_row)
        if n[0] == 1:
            answer[1] = 'N'
        else:
            answer[1] = 'S'

        t = self.t_classifier.predict(bow_row)
        if t[0] == 1:
            answer[2] = 'T'
        else:
            answer[2] = 'F'

        j = self.j_classifier.predict(bow_row)
        if j[0] == 1:
            answer[3] = 'J'
        else:
            answer[3] = 'P'

        return ''.join(answer)

model = SVMWithBagOfWords()
print(model.classify('  its not the end of the world please relax theres still music and art'))