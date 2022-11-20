
import matplotlib.pyplot as plt
import numpy as np
import pickle 
def plot_result(history):

    fig, (ax1,  ax2)= plt.subplots(1, 2, figsize=(7, 4))
    #fig.subplots_adjust(left=1, right=1.0, top=0.94, bottom=0.16)
    fig.subplots_adjust(top=0.75,  bottom=0.30)


    ax1.plot(history["acc"])
    ax1.plot(history["val_acc"])
    ax2.plot(history["loss"])
    ax2.plot(history["val_loss"])
    ax1.set_xlabel("Epochs", fontsize = 12)
    ax2.set_xlabel("Epochs", fontsize = 12)
    ax1.set_ylabel("Accuracy", fontsize = 12)
    ax2.set_ylabel("Loss", fontsize = 12)
    ax1.set_title("Train/Validation Accuracy with ResNet50 Transfer Learning", fontsize = 13)
    ax2.set_title("Train/Validation Loss with ResNet50 Transfer Learning", fontsize = 13)
    ax1.legend(["train", "validation"], loc=2)
    ax2.legend(["train", "validation"], loc=1)
    #plt.tight_layout()

    plt.show()

with open('/processing/ertugrul/Part_2/TL_Resnet/TL_Resnet2.pkl', 'rb') as file:
      
    # Call load method to deserialze
    history = pickle.load(file)
    
plot_result(history)