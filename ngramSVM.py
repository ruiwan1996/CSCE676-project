import re
import pandas as pd
from sklearn.svm import SVC
from scipy.sparse import coo_matrix
from get_ngrams import preprocess_post, get_ngrams

class SVMWithNgrams:
    def __init__(self):

        ngrams, ngram_user = get_ngrams(2)
        df = pd.read_csv('mbti_1.csv')
        mbti_type_series = df['type']

        self.ngrams = ngrams
        self.i_classifier = None
        self.n_classifier = None
        self.t_classifier = None
        self.j_classifier = None

        self.__train__(mbti_type_series, ngram_user)

    def __train__(self, mbti_type_series, ngram_user):
        y = mbti_type_series

        print("1")
        self.i_classifier = SVC()
        self.i_classifier.fit(ngram_user, y.apply(lambda x: x[0] == 'I'))

        print("2")
        self.n_classifier = SVC()
        self.n_classifier.fit(ngram_user, y.apply(lambda x: x[1] == 'N'))

        print("3")
        self.t_classifier = SVC()
        self.t_classifier.fit(ngram_user, y.apply(lambda x: x[2] == 'T'))

        print("4")
        self.j_classifier = SVC()
        self.j_classifier.fit(ngram_user, y.apply(lambda x: x[3] == 'J'))

    def classify(self, post):
        post = preprocess_post(post)
        bow_row = [len(re.findall(ngram, post)) for ngram in self.ngrams]
        bow_row = coo_matrix(bow_row, shape=(1, len(bow_row)))

        answer = [None, None, None, None]

        i = self.i_classifier.predict(bow_row)
        print(i[0])
        if i[0] == 1:
            answer[0] = 'I'
        else:
            answer[0] = 'E'

        n = self.n_classifier.predict(bow_row)
        print(n[0])
        if n[0] == 1:
            answer[1] = 'N'
        else:
            answer[1] = 'S'

        t = self.t_classifier.predict(bow_row)
        print(t[0])
        if t[0] == 1:
            answer[2] = 'T'
        else:
            answer[2] = 'F'

        j = self.j_classifier.predict(bow_row)
        print(j[0])
        if j[0] == 1:
            answer[3] = 'J'
        else:
            answer[3] = 'P'

        return ''.join(answer)

#model = SVMWithNgrams()
#print(model.classify('  its not the end of the world please relax theres still music and art'))