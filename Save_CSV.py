class save_to_csv:

    def __init__(self, test_label):
        self.test_label = test_label

    def save(self, pred, fileName):
        classification_result = []
        for i in range(len(self.test_label)):
          for j in range(len(self.test_label[i]['sentiment'])):
              if self.test_label[i]['sentiment'].iloc[j] == 0:
                  if pred[i][j] == 0:
                      classification_result.append(str(j+1) + ", N, N")
                  elif pred[i][j] == 1:
                      classification_result.append(str(j+1) + ", N, P")
              elif self.test_label[i]['sentiment'].iloc[j] == 1:
                  if pred[i][j] == 0:
                      classification_result.append(str(j+1) + ", P, N")
                  elif pred[i][j] == 1:
                      classification_result.append(str(j+1) + ", P, P")

        labels = ["data ke-i", "actual", "predict"]
        # Open File
        output_file = open(fileName+".csv", 'w')
        output_file.write(','.join(labels) + "\n")
        # Write data to file
        for r in classification_result:
            output_file.write(r + "\n")
        output_file.close()
        print("Hasil klasifikasi berhasil disimpan ke dalam file .csv!")