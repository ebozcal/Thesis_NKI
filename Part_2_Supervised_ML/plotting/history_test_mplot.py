
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pickle
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


#with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/ResNet10_history_2ch_crp_762.pkl"), 'rb') as f:
 #   history = pickle.load(f)
with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/final_test/CNN_SGD_762_2ch_history_cropped_10next.pkl"), 'rb') as f:
    history = pickle.load(f)
    
    
fig, (ax1,  ax2)= plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(savgol_filter(history["acc"][2:], 17, 1))
ax1.plot(savgol_filter(history["val_acc"][2:],17,1))
ax2.plot(savgol_filter(history["loss"][2:], 17, 1))
ax2.plot(savgol_filter(history["val_loss"][2:], 17, 1))
ax1.set_xlabel("Epochs", fontsize = 12)
ax2.set_xlabel("Epochs", fontsize = 12)
ax1.set_ylabel("Accuracy", fontsize = 12)
ax2.set_ylabel("Loss", fontsize = 12)
ax1.set_title("Accuracy for CNN 2 ch masked 10 next")
ax2.set_title("Loss for CNN 2ch masked 10 next")
ax1.legend(["train", "validation"], loc=4, fontsize = 12)
ax2.legend(["train", "validation"], loc=1, fontsize = 12)
plt.tight_layout()

plt.show()


