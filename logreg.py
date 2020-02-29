from sklearn.linear_model import LogisticRegression
from time import time

class logreg:

    def __init__(self, train, train_label, test, test_label):
        self.train = train
        self.train_label = train_label
        self.test = test
        self.test_label = test_label

    def logreg(self):
        pred_res_per_fold = []
        k = 1
        for i in range(len(self.train)):
            print("Fold ", k)
            print("Pengklasifikasian menggunakan Logistic Regression...")
            start = time()

            logreg = LogisticRegression()
            logreg.fit(self.train[i], self.train_label[i]['sentiment'])

            outcome = logreg.predict(self.test[i])
            end = time()
            print("Selesai.")
            print("Waktu Komputasi: ", str(round((end - start), 3)), "detik")
            print("=====================================\n")
            pred_res_per_fold.append(outcome)
            k += 1

        return pred_res_per_fold
