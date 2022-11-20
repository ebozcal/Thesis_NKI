import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import scikitplot as skplt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle


with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/ResNet_label__pred_cropped_20.pkl"), 'rb') as f:
    dict = pickle.load(f)
#with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/final_test/REsNet50_TL_history_FT_label__pred.pkl"), 'rb') as f:
 #   dict = pickle.load(f)
y_true2 = dict["label"]
y_pred2 = dict["pred"]
target_names = ["Stable", "Response", "Progres"]
print(classification_report(y_true2, y_pred2, target_names = target_names))
y_true = label_binarize(y_true2, classes=[0, 1, 2])
y_pred = label_binarize(y_pred2, classes=[0, 1, 2])

# Learn to predict each class against the other


n_classes = 3 # number of class



lw=2
# Compute ROC curve and ROC area for each class
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i], )
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
roc_auc_scores = []
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
classes = ["SD", "PR", "PD"]
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(classes[i], roc_auc[i]), fontsize = 12)
    roc_auc_scores.append(roc_auc[i])

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 14)
plt.ylabel('True Positive Rate', fontsize = 14)
plt.title('ResNet ROC Curve', fontsize = 14)
plt.legend(loc="lower right")
plt.show()