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

def plot_learning_curve_train_size(result_dir, kfold=10, datasize=160):
    result_dirnames = [i for i in os.listdir(result_dir) if i.endswith("{:d}fold".format(kfold))]
    result_dirnames.sort()
    train_sizes, APs, AUC_ROCs = [], [], []
    for dirname in result_dirnames:
        test_ratio = float(dirname.split("testSize")[0])
        train_size = datasize * (kfold - 1) / kfold * (1 - test_ratio)
        result_path = os.path.join(result_dir, dirname, "results.csv")
        result = pd.read_csv(result_path).values[0]
        train_sizes.append(train_size)
        AUC_ROCs.append(result[0])
        APs.append(result[1])

    fig, axes = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8))
    axes[0].plot(train_sizes, AUC_ROCs, marker=".")
    axes[0].grid()
    axes[0].set_title("AUC_ROC learning curve")
    axes[1].plot(train_sizes, APs, marker=".")
    axes[1].grid()
    axes[1].set_title("APs learning curve")
    plt.savefig(os.path.join(result_dir, "learning_curve.png"), bbox_inches="tight")
    plt.show()
    print("")

if __name__ == '__main__':
    plot_learning_curve_train_size("/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/results/LearnCurve_luna_160", datasize=160)