
import matplotlib.pyplot as plt

def plot_result(history):

    fig, (ax1,  ax2)= plt.subplots(1, 2, figsize=(7, 4))
    ax1.plot(history.history["acc"])
    ax1.plot(history.history["val_acc"])
    ax2.plot(history.history["loss"])
    ax2.plot(history.history["val_loss"])
    ax1.set_xlabel("Epochs", fontsize = 12)
    ax2.set_xlabel("Epochs", fontsize = 12)
    ax1.set_ylabel("Accuracy", fontsize = 12)
    ax2.set_ylabel("Loss", fontsize = 12)
    ax1.set_title("Accuracy with Epochs Whole CT 8 batch Lb2_20")
    ax2.set_title("Loss with Epochs Whole CT 8 batch Lb2_20")
    ax1.legend(["train", "validation"], loc=2, fontsize = 12)
    ax2.legend(["train", "validation"], loc=1, fonttsize = 12)
    plt.tight_layout()

    plt.show()

