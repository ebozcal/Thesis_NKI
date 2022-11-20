import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


#with open((), 'rb') as f:
 #   dict = pickle.load(f)
with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/final_test/CNN_SGD_762_2ch_label__pred_cropped_10next.pkl"), 'rb') as f:
    dict = pickle.load(f)
    
y_true = dict["label"]
y_pred = dict["pred"]
print(y_pred)
print(y_true)

print("accuracy:",accuracy_score(y_true, y_pred) )
classes = ["Stable", "Response", "Progress"]
cmtest = confusion_matrix(y_true, y_pred)

fig, ax1 = plt.subplots(figsize=(6, 6))
ax1.matshow(cmtest, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cmtest.shape[0]):
    for j in range(cmtest.shape[1]):
        ax1.text(x=j, y=i,s=cmtest[i, j], va='center', ha='center', size='large')

 
plt.xlabel('Actuals', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.title('ResNet50_TL Test Set Confusion Matrix', fontsize=18)
plt.xticks(range(3))
plt.xticks(range(3))
plt.show()