from sklearn.svm import LinearSVC
from time import time

class SVM:
    def __init__(self, train, train_label, test, test_label):
        self.train = train
        self.train_label = train_label
        self.test = test
        self.test_label = test_label

    def svm(self):
        pred_res_per_fold = []
        k = 1
        for i in range(len(self.train)):
            print("Fold ", k)
            print("Pengklasifikasian menggunakan SVM...")
            start = time()

            svm = LinearSVC()
            svm.fit(self.train[i], self.train_label[i]['sentiment'])

            outcome = svm.predict(self.test[i])
            end = time()
            print("Selesai.")
            print("Waktu Komputasi: ", str(round((end - start), 3)), "detik")
            print("=====================================\n")
            pred_res_per_fold.append(outcome)
            k += 1

        return pred_res_per_fold