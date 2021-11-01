from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from utils.summary_utils import Logger

import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os



def test(model_dir, model_name, kfold=5):

    labels_test = []
    probs_test = []
    if kfold is None or kfold == 1:
        temp = np.load(os.path.join(model_dir, "preds.npz"), allow_pickle=True)
        l, p = temp["l"], temp["p"].squeeze()
        labels_test.append(l)
        probs_test.append(p)
    else:
        for i in range(kfold):
            split_dir = "{:s}/{:s}.kfold{:d}".format(model_dir, model_name, i)
            temp = np.load(os.path.join(split_dir, "preds.npz"), allow_pickle=True)
            l, p = temp["l"], temp["p"].squeeze()
            labels_test.append(l)
            probs_test.append(p)
    labels_test = np.concatenate(labels_test)
    probs_test = np.concatenate(probs_test)
    # np.savez_compressed(os.path.join(model_dir, "preds.npz"),
    #                     l=labels_test, p=probs_test)
    # all_label = np.argmax(labels_test, axis=-1)
    # all_pred = np.argmax(probs_test, axis=-1)
    all_label = labels_test
    all_pred = np.array(probs_test > 0.5).astype(np.int)

    all_label = 1 - all_label
    all_pred = 1 - all_pred
    labels_test = 1 - labels_test
    probs_test = 1 - probs_test



    os.makedirs(model_dir, exist_ok=True)
    # fpr, tpr, ths = roc_curve(labels_test[:, 0], probs_test[:, 0])
    fpr, tpr, ths = roc_curve(1-labels_test, 1-probs_test)
    # roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score(1-labels_test, 1-probs_test)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_dir, "roc_curve.png"), bbox_inches="tight", dpi=200)
    plt.close()

    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = ths[optimal_idx]

    confMat = confusion_matrix(all_label, all_pred)
    df_cm = pd.DataFrame(confMat, index=["maligant", "benign"],
                         columns=["maligant", "benign"])
    print("\n ---------- Threshold is {:f} ---------- ".format(0.5))
    print("Classification report with th_{:f}: ".format(0.5))
    print(classification_report(all_label, all_pred))
    print("Test confusion matrix with th_{:f}: ".format(0.5))
    print(df_cm)
    plt.figure()
    # sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
    sns.heatmap(df_cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")
    plt.savefig(os.path.join(model_dir, "test_confusion_matrix_th_0.5.png"), bbox_inches="tight", dpi=200)
    plt.close()


    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = ths[optimal_idx]
    # all_pred_new = 1 - (probs_test[:, 0] >= optimal_threshold).astype(np.int)
    all_pred_new = 1 - ((1 - probs_test) >= optimal_threshold).astype(np.int)
    confMat = confusion_matrix(all_label, all_pred_new)
    df_cm = pd.DataFrame(confMat, index=["maligant", "benign"],
                         columns=["maligant", "benign"])
    print("\n ---------- Threshold is {:f} ---------- ".format(optimal_threshold))
    print("Classification report with th_{:f}: ".format(optimal_threshold))
    print(classification_report(all_label, all_pred_new))
    print("Test confusion matrix with th_{:f}: ".format(optimal_threshold))
    print(df_cm)
    plt.figure()
    # sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
    sns.heatmap(df_cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")
    plt.savefig(os.path.join(model_dir, "test_confusion_matrix_th_{:.3f}.png".format(optimal_threshold)),
                bbox_inches="tight", dpi=200)
    plt.close()


    select_th = ths[np.argmax(tpr >= 0.8)]
    all_pred_new = 1 - ((1 - probs_test) >= select_th).astype(np.int)
    # all_pred_new = 1 - (probs_test[:, 0] >= select_th).astype(np.int)
    confMat = confusion_matrix(all_label, all_pred_new)
    df_cm = pd.DataFrame(confMat, index=["maligant", "benign"],
                         columns=["maligant", "benign"])
    print("\n ---------- Threshold is {:f} ---------- ".format(select_th))
    print("Classification report with th_{:f}: ".format(select_th))
    print(classification_report(all_label, all_pred_new))
    print("Test confusion matrix with th_{:f}: ".format(select_th))
    print(df_cm)
    plt.figure()
    # sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
    sns.heatmap(df_cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")
    plt.savefig(os.path.join(model_dir, "test_confusion_matrix_th_{:.3f}.png".format(select_th)),
                bbox_inches="tight", dpi=200)
    plt.close()


    AP = average_precision_score(1 - labels_test, 1 - probs_test)
    precision, recall, thresholds = precision_recall_curve(1 - labels_test, 1 - probs_test)
    plt.figure()
    plt.plot(recall, precision, label='precision-recall curve (AP = %0.2f)' % AP)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(model_dir, "precision_recall_curve.png"), bbox_inches="tight", dpi=200)
    plt.close()

    with open(os.path.join(model_dir, "results.csv"), "w") as f:
        f.write("AUC-ROC, AP\n")
        f.write("{:f}, {:f}".format(roc_auc, AP))

def main():

    log_file = "eval.log"
    log_path = os.path.join(args.save_dir, log_file)
    sys.stdout = Logger(log_path)
    print("=" * 100)
    print("Working in directory: {:s}\n".format(args.save_dir))

    test(args.save_dir, args.model, args.kfold)

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--datasource', type=str, help='SEAMII, BP2004', default="SEAMII")
    parser.add_argument('--save_dir', type=str, help="directory of saved results", default="AACR_results/bs_16.lr_0.001.beta_0.001.aug.balanceAfterSplit")
    parser.add_argument('--kfold', type=int, help='number of kfold', default=5)
    parser.add_argument("-m", "--model", default="res18", help="model")
    # parser.add_argument('--epochs', type=int, help='number of epochs', default=50)
    # parser.add_argument('--batchsize', type=int, help='batch size', default=16)
    # parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    # parser.add_argument('--image_size', type=int, help='image size', default=224)
    # parser.add_argument('--num_classes', type=int, help='number of classes', default=2)
    # parser.add_argument('--l2norm_beta', type=float, help='beta for l2 norm on weights', default=0.001)
    # parser.add_argument('--train', type=eval, help='train or test', default=False)
    # parser.add_argument('--load_pretrain', type=eval, help='whether to load pretrained model on imagenet', default=True)
    # parser.add_argument('--augmentation', type=eval, help='whether to use image augmentation', default=True)
    # parser.add_argument('--balance_option', type=str, help='before or after train_test_split',
    #                     choices=["before", "after"], default="after")
    parser.add_argument('--gpu', type=str, help='which gpu to use', default="7")
    # parser.add_argument('--kfold', type=int, help='number of kfold', default=None)
    # parser.add_argument('--splitId', type=int, help='kfold split idx', default=None)
    # parser.add_argument('--load_model', type=str, help='trained model to load',
    #                     # default="/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/privacy/VGG2D/results/bs_16.lr_0.001_copy/vgg19_epoch22.npy")
    #                     # default="/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/privacy/VGG2D/results/bs_16.lr_0.001.beta_0.001.aug.balanceBeforeSplit.best/vgg19_epoch37.npy")
    #                     # default="/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/privacy/VGG2D/results/bs_16.lr_0.001.beta_0.001.aug.balanceBeforeSplit/vgg19_epoch38.npy")
    #                     # default="/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/privacy/VGG2D/results/bs_16.lr_0.001.beta_0.001.aug.balanceAfterSplit.best/vgg19_epoch40.npy")
    #                     # default="/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/privacy/VGG2D/results/bs_16.lr_0.001.beta_0.001.aug.balanceAfterSplit/vgg19_epoch37.npy")
    #                     # default="/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/privacy/VGG2D/results/bs_16.lr_0.001.beta_0.001.aug.balanceAfterSplit.exp2/vgg19_epoch24.npy")
    #                     default=None)
    # parser.add_argument('--extraStr', type=str, help='extraStr for saving', default="")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main()