from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_learning_cuvre(train_loss, test_loss, train_acc, test_acc, save_dir, epochs):
    plt.figure()
    plt.plot(train_loss, label="Train loss")
    plt.plot(test_loss, label="Test loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plot_name = "loss curve_ep{:d}.png".format(epochs)
    plt.savefig(os.path.join(save_dir, plot_name), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(train_acc, label="Train acc")
    plt.plot(test_acc, label="Test acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plot_name = "acc curve_ep{:d}.png".format(epochs)
    plt.savefig(os.path.join(save_dir, plot_name), dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(plot_folder, file_name, all_labels, all_preds, epoch, num_classes):
    confMat = confusion_matrix(all_labels, all_preds, np.arange(num_classes))
    df_cm = pd.DataFrame(confMat, index=[i for i in range(num_classes)],
                         columns=[i for i in range(num_classes)])
    print("{:s}:".format(file_name))
    print(df_cm)
    fig = plt.figure()
    sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
    plt.savefig(os.path.join(plot_folder, "{:s}_ep{:d}.png".format(file_name, epoch)), bbox_inches="tight", dpi=200)
    plt.close()
    return fig