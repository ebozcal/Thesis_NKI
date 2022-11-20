
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pickle
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


#with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/CNNclass_history_cropped_10.pkl"), 'rb') as f:
#    historyCNN_crp_10 = pickle.load(f)


with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/CNNclass_history_masked_20.pkl"), 'rb') as f:
    historyCNN_masked_20 = pickle.load(f)



#with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/CNNclass_history_cropped_20.pkl"), 'rb') as f:
 #   historyCNN_crp_20 = pickle.load(f)


with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/ResNet_history_cropped_20.pkl"), 'rb') as f:
    historyresnet_crp_20 = pickle.load(f)

with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/ResNet50_history_masked_20.pkl"), 'rb') as f:
    historyresnetmasked20 = pickle.load(f)

with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/ResNet10_history_cropped_762.pkl"), 'rb') as f:
    historyresnetcrp_10_next = pickle.load(f)

with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/ResNet50_history_2ch_masked_762.pkl"), 'rb') as f:
    historyresnet_2chmasked_10_next = pickle.load(f)

with open(("/processing/ertugrul/Part_2/plot_test_history_hamilton/history_plot/results/final_test/CNN_SGD_762_masked_history_cropped_10next.pkl"), 'rb') as f:
    historyCNN_masked_10next = pickle.load(f)
    
fig, (ax1,  ax2)= plt.subplots(1, 2, figsize=(10, 4))

#ax1.plot(savgol_filter(history10["loss"][14:], 17, 1))
ax1.plot(savgol_filter(historyCNN_masked_20["loss"][14:150], 37, 1))
#ax1.plot(savgol_filter(historyCNN_crp_20["loss"][14:], 17, 1))
ax1.plot(savgol_filter(historyresnet_crp_20["loss"][14:150], 17, 1))
ax1.plot(savgol_filter(historyresnetmasked20["loss"][14:], 17, 1))
ax1.plot(savgol_filter(historyresnetcrp_10_next["loss"][14:], 17, 1))
ax1.plot(savgol_filter(historyresnet_2chmasked_10_next["loss"][14:], 17, 1))
ax1.plot(savgol_filter(historyCNN_masked_10next["acc"][14:], 17, 1))




#ax1.plot(history10["val_acc"][2:])
#ax1.plot(history20masked["val_acc"][2:])

#ax2.plot(savgol_filter(history10["val_loss"][14:], 17, 1))
ax2.plot(savgol_filter(historyCNN_masked_20["val_loss"][14:150], 37, 1))
#ax2.plot(savgol_filter(historyCNN_crp_20["val_loss"][14:], 17, 1))
ax2.plot(savgol_filter(historyresnet_crp_20["val_loss"][14:150], 17, 1))
ax2.plot(savgol_filter(historyresnetmasked20["val_loss"][14:], 17, 1))
ax2.plot(savgol_filter(historyresnetcrp_10_next["val_loss"][14:], 17, 1))
ax2.plot(savgol_filter(historyresnet_2chmasked_10_next["val_loss"][14:], 17, 1))
ax2.plot(savgol_filter(historyCNN_masked_10next["val_loss"][14:], 17, 1))





#ax2.plot(history10["val_loss"][2:])
#ax2.plot(history20masked["val_loss"][2:])

ax1.set_xlabel("Epochs", fontsize = 12)
ax2.set_xlabel("Epochs", fontsize = 12)
ax1.set_ylabel("Loss", fontsize = 12)
ax2.set_ylabel("Loss", fontsize = 12)
ax1.set_title("Training loss with different labels and data")
ax2.set_title("Validation loss with different labels and data")
ax1.legend(["CNNlb20&masked",  "RNetlb20&cropped",  "RNetlb20&masked", "RNetlb10next&cropped", "RNetlb10next&2chmasked", "CNNlb10next&masked"], loc=3, fontsize = 10)
ax2.legend(["CNNlb20&masked",  "RNetlb20&cropped", "RNetlb20&masked", "RNetlb10next&cropped", "RNetlb10next&2chmasked", "CNNlb10next&masked"], loc=2, fontsize = 10)
plt.tight_layout()
plt.show()


