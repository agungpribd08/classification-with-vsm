from bs4 import BeautifulSoup
import re

class praproses:

    def __init__(self, dataset):
        self.dataset = dataset

    def load_stopwords(self):
        stops = open("stopwords.txt").read().split('\n')
        stopwords = set(stops)
        return stopwords

    def review_to_words(self, raw_review):
        review_text = BeautifulSoup(raw_review, "lxml").get_text()

        letters_only = re.sub("[^a-zA-Z]", " ", review_text)

        words = letters_only.lower().split()

        stopwords = self.load_stopwords()

        meaningful_words = [w for w in words if not w in stopwords]

        return (" ".join(meaningful_words))

    def cleaning(self):
        num_reviews = self.dataset.size
        clean_reviews = []
        for i in range(num_reviews):
            clean_reviews.append(self.review_to_words(self.dataset[i]))
        print("Tahapan praproses selesai.\n")
        return (clean_reviews)