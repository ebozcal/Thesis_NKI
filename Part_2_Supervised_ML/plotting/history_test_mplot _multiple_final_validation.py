
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pickle
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/final_test/CNNclass_762_history_cropped_10next.pkl"), 'rb') as f:
    historycnn = pickle.load(f)


with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/REsNet50_672_crp_TL_history_FT.pkl"), 'rb') as f:
    historyresnet = pickle.load(f)



with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/final_test/REsNet50_TL_history_FT.pkl"), 'rb') as f:
    historyresnetTL = pickle.load(f)



fig, (ax1,  ax2)= plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(savgol_filter(historycnn["val_loss"][2:], 17, 1))
ax1.plot(savgol_filter(historyresnet["val_loss"][2:], 17, 1))
ax1.plot(savgol_filter(historyresnetTL["val_loss"][2:], 17, 1))




ax2.plot(savgol_filter(historycnn["val_acc"][29:], 17, 1))
ax2.plot(savgol_filter(historyresnet["val_acc"][29:], 17, 1))
ax2.plot(savgol_filter(historyresnetTL["val_acc"][29:], 17, 1))


#ax1.plot(history10["val_acc"][2:])
#ax1.plot(history20masked["val_acc"][2:])



#ax2.plot(history10["val_loss"][2:])
#ax2.plot(history20masked["val_loss"][2:])

ax1.set_xlabel("Epochs", fontsize = 12)
ax2.set_xlabel("Epochs", fontsize = 12)
ax1.set_ylabel("Loss", fontsize = 12)
ax2.set_ylabel("Accuracy", fontsize = 12)
ax1.set_title("Validation loss for 3 Models")
ax2.set_title("Validation accuracy for 3 Models")
ax1.legend(["CNN", "ResNet50", "ResNet50TL"], loc=1, fontsize = 10)
ax2.legend(["CNN", "ResNet50", "ResNet50TL"], loc=4, fontsize = 10)
plt.tight_layout()

plt.show()


