import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import scikitplot as skplt
from itertools import cycle
from sklearn.preprocessing import label_binarize


with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/final_test/CNN_SGD_762_2ch_label__pred_cropped_10next.pkl"), 'rb') as f:
    dict = pickle.load(f)
#with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/REsNet50_762_TL_Adam_crp_label_pred.pkl"), 'rb') as f:
 #   dict = pickle.load(f)
y_true2 = dict["label"]
y_pred2 = dict["pred"]
target_names = ["Stable", "Response", "Progres"]
print(classification_report(y_true2, y_pred2, target_names = target_names))
y_true2 = label_binarize(y_true2, classes=[0, 1, 2])
y_pred2 = label_binarize(y_pred2, classes=[0, 1, 2])
print("roc_auc_score:", roc_auc_score(y_true2, y_pred2, average = "macro", multi_class ="ovr"))

n_classes = y_true2.shape[1]

fpr2 = {}
tpr2 = {}
roc_auc2 = {}
for i in range(n_classes):
    fpr2[i], tpr2[i], _ = roc_curve(y_true2[:, i], y_pred2[:, i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])
print(fpr2)
# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
fpr2["macro"], tpr2["macro"], _ = roc_curve(y_true2.ravel(), y_pred2.ravel())
roc_auc2["macro"] = auc(fpr2["macro"], tpr2["macro"])

plt.figure()
lw = 2
plt.plot(fpr2[0], tpr2[0], color="green", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc2[0],)
plt.plot(fpr2[1], tpr2[1], color="red", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc2[1],)
plt.plot(fpr2[2], tpr2[2], color="blue",  lw=lw,  label="ROC curve (area = %0.2f)" % roc_auc2[2],)
plt.plot([0, 1], [0, 1], color="yellow", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize = 12)
plt.ylabel("True Positive Rate", fontsize = 12)
plt.title("ROC Curve for CNN", fontsize = 14)
plt.legend(["SD vs rest", "PR vs rest", "PD vs rest"], loc="lower right", fontsize = 12)
plt.show()
