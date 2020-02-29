from scipy import interp
from sklearn.metrics import roc_curve, auc
from matplotlib.figure import Figure
import numpy as np

class aucroc:
    def __init__(self, test_label, predict_result):
        self.test_label = test_label
        self.predict_result = predict_result

    def plot_auroc(self, roc_name):
        fig1 = Figure(figsize=(8, 8), dpi=100)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        i = 1
        for j in range(len(self.test_label)):
            fpr, tpr, t = roc_curve(self.test_label[j]['sentiment'], self.predict_result[j])
            tprs.append(interp(mean_fpr, fpr, tpr))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            fig1.add_subplot(111).plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.5f)' % (i, roc_auc))
            i = i + 1

        fig1.add_subplot(111).plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        fig1.add_subplot(111).plot(mean_fpr, mean_tpr, color='blue',
                 label=r'Mean ROC (AUC = %0.5f )' % (mean_auc), lw=2, alpha=1)

        fig1.add_subplot(111).set_xlabel('False Positive Rate')
        fig1.add_subplot(111).set_ylabel('True Positive Rate')
        fig1.add_subplot(111).set_title(roc_name)
        fig1.add_subplot(111).legend(loc="lower right")

        return fig1, aucs