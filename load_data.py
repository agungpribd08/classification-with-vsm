import pandas as pd

class load_dataset:

    def __init__(self, file):
        self.file = file

    def load(self):
        df = pd.read_csv(self.file, header=0, \
                         delimiter="\t", quoting=3)
        df = df[['id', 'sentiment', 'review']]
        df = df[pd.notnull(df['review'])]
        print("File berhasil dimuat.")
        return (df)
