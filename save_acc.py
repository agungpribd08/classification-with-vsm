class save_acc:

    def __init__(self, auc):
        self.auc = auc

    def save(self, fileName):
        result = []
        for i in range(len(self.auc)):
            result.append(str(i+1) +","+ str(self.auc[i]))
        labels = ["fold", "akurasi"]
        print("Creating CSV file..")
        # Open File
        output_file = open(fileName + ".csv", 'w')
        output_file.write(','.join(labels) + "\n")
        # Write data to file
        for r in result:
            output_file.write(r + "\n")
        output_file.close()
        print("File saved!")
