import multiprocessing
from gensim.models import Doc2Vec
from tqdm import tqdm
from sklearn import utils
import nltk
from gensim.models.doc2vec import TaggedDocument
#nltk.download('punkt')

class d2v:

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def tokenize_text(self, text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) < 2:
                    continue
                tokens.append(word.lower())
        return tokens

    def start_tag(self):
        train_tagged = self.train.apply(
            lambda r: TaggedDocument(words=self.tokenize_text(r['review']), tags=[r.sentiment]), axis=1)
        test_tagged = self.test.apply(
            lambda r: TaggedDocument(words=self.tokenize_text(r['review']), tags=[r.sentiment]), axis=1)

        return train_tagged, test_tagged

    def create_model(self, train_tagged):
        cores = multiprocessing.cpu_count()
        model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=1, window=15, sample=0, workers=cores)
        model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

        for epoch in range(30):
            model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]),
                             total_examples=len(train_tagged.values), epochs=1)
            model_dbow.alpha -= 0.002
            model_dbow.min_alpha = model_dbow.alpha

        return model_dbow

    def vec_for_learning(self, model, tagged_docs):
        sents = tagged_docs.values
        targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])

        return targets, regressors

    def d2v(self, k):
        print("Pembentukan Doc2Vec pada fold-", k)
        train_tagged, test_tagged = self.start_tag()
        model_dbow = self.create_model(train_tagged)
        y_train, X_train = self.vec_for_learning(model_dbow, train_tagged)
        y_test, X_test = self.vec_for_learning(model_dbow, test_tagged)
        print("Selesai.")
        print("=====================================\n")

        return X_train, X_test
