from time import time
from sklearn.feature_extraction.text import TfidfVectorizer

class BagOfWords:

    nsubset = 500

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def bow_tfidf(self):

        bow_tfidf_train = []
        bow_tfidf_test = []
        k = 1
        for i in range(len(self.train)):
            print("Pembentukan Bag Of Words pada fold-", k)
            start = time()
            tf_transformer = TfidfVectorizer(max_features=5000)
            bow_tfidf_train.append(tf_transformer.fit_transform(self.train[i]['review']))
            bow_tfidf_test.append(tf_transformer.transform(self.test[i]['review']))
            end = time()
            print("Time: ", str(round((end - start), 2)), "seconds")
            k = k + 1
            print("Selesai.")
            print("=====================================\n")

        return bow_tfidf_train, bow_tfidf_test
