import nltk
from gensim.models.doc2vec import TaggedDocument
nltk.download('punkt')

class tagged_doc:
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
        train_tagged = self.train[0].apply(
            lambda r: TaggedDocument(words=self.tokenize_text(r['review']), tags=[r.sentiment]), axis=1)
        test_tagged = self.test[0].apply(
            lambda r: TaggedDocument(words=self.tokenize_text(r['review']), tags=[r.sentiment]), axis=1)

        return train_tagged, test_tagged