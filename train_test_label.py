class fold:

    def __init__(self, df):
        self.df = df

    def train_test(self):
        test = []
        train = []
        subset = 500
        j = 0
        for i in range(10):
            test.append(self.df.iloc[j:subset])
            j = j + 500
            subset = subset + 500
            train.append(self.df.drop(test[i].index))

        return train, test