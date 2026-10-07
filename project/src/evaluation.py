import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize




class Evaluator:
    def __init__(self, model, Xtest, ytest):
        self.model = model
        self.Xtest = Xtest
        self.ytest = ytest

        self.ypred = model.predict(Xtest)

        self.yproba = model.predict_proba(Xtest) if hasattr(model, "predict_proba") else None


    def report(self):
        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report(self.ytest, self.ypred))
        return self


    def confusion(self):
        print("\n=== CONFUSION MATRIX ===")
        cm = confusion_matrix(self.ytest, self.ypred)

        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        return self


def plot_roc_two_models(model1, model2, Xtest, ytest, labels=None):

    if labels is None:
        labels = sorted(set(ytest))

    # Binarizzazione delle classi
    y_bin = label_binarize(ytest, classes=labels)

    # Probabilità dei modelli
    y_proba1 = model1.predict_proba(Xtest)
    y_proba2 = model2.predict_proba(Xtest)

    # ROC micro-average
    fpr1, tpr1, _ = roc_curve(y_bin.ravel(), y_proba1.ravel())
    fpr2, tpr2, _ = roc_curve(y_bin.ravel(), y_proba2.ravel())

    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr2, tpr2)

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(fpr1, tpr1, label=f"Model 1 (AUC = {auc1:.3f})", linewidth=2)
    plt.plot(fpr2, tpr2, label=f"Model 2 (AUC = {auc2:.3f})", linewidth=2)

    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Micro Average (Multiclass)")
    plt.legend()
    plt.grid(True)
    plt.show()

