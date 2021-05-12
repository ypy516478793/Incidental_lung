"""
Instructions:
    python classify.py -s=results/DEBUG
"""

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from dataLoader.LUNA16Data import LUNA16
from tqdm import tqdm

from utils.summary_utils import Logger
from datetime import datetime
from classifier.resnet import generate_model

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import copy
import sys
import os

import argparse

parser = argparse.ArgumentParser(description="Incidental lung nodule classification")
parser.add_argument("-d", "--datasource", type=str, default="methodist", help="Dataset used for training/test",
                    choices=["luna", "lunaRaw", "methoidstPilot", "methodist", "additional"])
# parser.add_argument("-r", "--root_dir", type=str, help="Root directory for the dataset")
# parser.add_argument("-pp", "--pos_label_path", type=str, help="Position label file path")
# parser.add_argument("-cp", "--cat_label_path", type=str, help="Category label file path")
parser.add_argument("-s", "--save_dir", type=str, help="Save directory")
parser.add_argument("-g", "--gpu", type=str, default="0,1,2,3", help="Which gpus to use")

parser.add_argument("-m", "--model", default="res18", help="model")
parser.add_argument("-lm", "--load_model", type=str, default=None, help="Path/Directory of the model to be loaded")
parser.add_argument("-j", "--workers", default=0, type=int, help="number of data loading workers (default: 32)")
parser.add_argument("-t", "--train", type=eval, default=True, help="Train phase: True or False")
# parser.add_argument("-cl", "--clinical", type=eval, default=False, help="Whether to use clinical features")

parser.add_argument("-e", "--epochs", default=10, type=int, help="number of total epochs to run")
parser.add_argument("-se", "--start-epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("-sl", "--best_loss", default=np.inf, type=float, help="manual best loss (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=4, type=int, help="mini-batch size (default: 16)")
parser.add_argument("-lr", "--learning-rate", default=0.001, type=float, help="initial learning rate")
parser.add_argument("-mo", "--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("-wd", "--weight-decay", default=1e-3, type=float, help="weight decay (default: 1e-4)")

parser.add_argument("-es", "--extra_str", type=str, default="", help='extra string for data')

# parser.add_argument("-cs", "--cube_size", default=64, type=int, help="Cube size of the lung nodule")

# "flip": False, "swap": False, "scale": False, "rotate": False
parser.add_argument("--mask", default=True, type=eval, help="mask lung")
parser.add_argument("--crop", default=True, type=eval, help="crop lung")

parser.add_argument("--flip", default=False, type=eval, help="flip")
parser.add_argument("--swap", default=False, type=eval, help="swap")
parser.add_argument("--scale", default=False, type=eval, help="scale")
parser.add_argument("--rotate", default=False, type=eval, help="rotate")
parser.add_argument("--contrast", default=False, type=eval, help="contrast")
parser.add_argument("--bright", default=False, type=eval, help="bright")
parser.add_argument("--sharp", default=False, type=eval, help="sharp")
parser.add_argument("--splice", default=False, type=eval, help="splice")

parser.add_argument("-k", "--kfold", default=None, type=int, help="number of kfold for train_val")
parser.add_argument("-ki", "--splitId", default=None, type=int, help="split id when use kfold")

parser.add_argument("--n_test", default=2, type=int, help="number of gpu for test")
parser.add_argument("--train_patience", type=int, default=10, help="If the validation loss does not decrease for this number of epochs, stop training")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu


from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
# from torch.autograd import Variable
# from torch.nn import DataParallel


# parser.add_argument("-ds", "--dataset_size", type=int, default=20000, help="Datasize, <= 20000")
# parser.add_argument("-en", "--encode", type=eval, default=True, help="Encode the image into categories")
# parser.add_argument("-se", "--sobel_edge", type=eval, default=True, help="Use sobel to detect edge first")
# parser.add_argument("-ce", "--combine_edge", type=eval, default=True, help="Combine detected edge")
# parser.add_argument("-lt", "--log_trans", type=eval, default=False, help="Use log transform or not")
# parser.add_argument("-e", "--epochs", type=int, default=25, help="Number of epochs for training")
# parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for training")
# parser.add_argument("-l", "--lr", type=float, default=0.001, help="Learning rate")
# parser.add_argument("-wd", "--weight_decay", type=float, default=0, help="Regularization weight")
# parser.add_argument("-o", "--optimizer", type=str, default="adam", help="Optimizer type", choices=["adam", "sgd"])
# parser.add_argument("-wr", "--workers", type=int, default=0, help="Number of workers")
# parser.add_argument("-t", "--train", type=eval, default=True, help="Train phase: True or False")
# parser.add_argument("-lm", "--load_model", type=str, default=None, help="Path/Directory of the model to be loaded")
# parser.add_argument("-k", "--task", type=str, default="classification", help="Task type",
#                     choices=["classification", "regression"])
# parser.add_argument("-m", "--model_name", type=str, default="resnet", help="Select the backbone for training",
#                     choices=['resnet', 'resnet50', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception'])
# args = parser.parse_args()



# parser = argparse.ArgumentParser(description="PyTorch DataBowl3 Detector")
# parser.add_argument("--datasource", "-d", type=str, default="luna",
#                     help="luna, lunaRaw, methoidstPilot, methodistFull, additional")
# parser.add_argument("--model", "-m", metavar="MODEL", default="res18", help="model")
# # parser.add_argument("--config", "-c", default="config_methodistFull", type=str)
# parser.add_argument("-j", "--workers", default=0, type=int, metavar="N",
#                     help="number of data loading workers (default: 32)")
#
# parser.add_argument("--save-freq", default="1", type=int, metavar="S",
#                     help="save frequency")
# parser.add_argument("--resume", "-re", default="../detector/resmodel/res18fd9020.ckpt", type=str, metavar="PATH",
# # parser.add_argument("--resume", default="../detector/results/res18-20201020-113114/030.ckpt",
# # parser.add_argument("--resume", default="../detector_ben/results/res18-20201202-112441/026.ckpt",
# # parser.add_argument("--resume", default="../detector_ben/results/res18-20201223-115306/038.ckpt",
# # parser.add_argument("--resume", default="../detector_ben/results/res18-20210106-112050_incidental/001.ckpt",
# #                     type=str, metavar="PATH",
#                     help="path to latest checkpoint (default: none)")
# parser.add_argument("--save-dir", "-s", default='', type=str, metavar="SAVE",
#                     help="directory to save checkpoint (default: none)")
# parser.add_argument("--test", "-t", default=True, type=eval, metavar="TEST",
#                     help="1 do test evaluation, 0 not")
# parser.add_argument("--inference", "-i", default=False, type=eval,
#                     help="True if run inference (no label) else False")
# parser.add_argument("--testthresh", default=-3, type=float,
#                     help="threshod for get pbb")
# parser.add_argument("--split", default=8, type=int, metavar="SPLIT",
#                     help="In the test phase, split the image to 8 parts")  # Split changed to 1 just to check.
# # parser.add_argument("--gpu", default="4, 5, 6, 7", type=str, metavar="N",
# parser.add_argument("--gpu", default="0, 1, 2, 3", type=str, metavar="N",
#                     help="use gpu")
# parser.add_argument("--rseed", default=None, type=int, metavar="N",
#                     help="random seed for train/val/test data split")
# parser.add_argument("--limit_train", default=None, type=float, metavar="N",
#                     help="ratio of training size")
#
# # "flip": False, "swap": False, "scale": False, "rotate": False
# parser.add_argument("--mask", default=True, type=eval, help="mask lung")
# parser.add_argument("--crop", default=True, type=eval, help="crop lung")
#
# parser.add_argument("--flip", default=False, type=eval, help="flip")
# parser.add_argument("--swap", default=False, type=eval, help="swap")
# parser.add_argument("--scale", default=False, type=eval, help="scale")
# parser.add_argument("--rotate", default=False, type=eval, help="rotate")
# parser.add_argument("--contrast", default=False, type=eval, help="contrast")
# parser.add_argument("--bright", default=False, type=eval, help="bright")
# parser.add_argument("--sharp", default=False, type=eval, help="sharp")
# parser.add_argument("--splice", default=False, type=eval, help="splice")
#
# parser.add_argument("--kfold", default=None, type=int, help="number of kfold for train_val")
# parser.add_argument("--split_id", default=None, type=int, help="split id when use kfold")
#
# parser.add_argument("--n_test", default=2, type=int, metavar="N",
#                     help="number of gpu for test")
# parser.add_argument("--train_patience", type=int, default=10,
#                     help="If the validation loss does not decrease for this number of epochs, stop training")



def train(trainLoader, testLoader, trainValLoader, model, optimizer, scheduler, criterion, save_dir,
          epochs=30, start_epoch=0):

    model_copy = copy.deepcopy(model)

    plot_folder = os.path.join(save_dir, "plots")
    os.makedirs(plot_folder, exist_ok=True)
    
    num_classes = 2
    best_val_loss = np.inf
    best_acc = 0
    best_epoch = epochs - 1

    # print_per_iteration = len(trainLoader) / 10
    print_per_iteration = 1
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    test_step = []
    for epoch in tqdm(range(start_epoch, epochs)):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)

        # Training phase
        model.train()
        all_preds = []
        all_probs = []
        all_labels = []

        running_loss = 0.0
        running_corrects = 0
        itr = 0
        s_time = time.time()

        for sample_batch in trainLoader:
            # inputs, labels = sample_batch["cubes"], sample_batch["label"]
            inputs, labels = sample_batch
            if isinstance(inputs, list):
                inputs = torch.from_numpy(np.array(inputs).astype(np.float32)).to(device)
                labels = torch.from_numpy(np.array(labels).astype(np.int)).to(device)
            else:
                inputs = inputs.float().to(device)
                labels = labels.to(device)

            # cube_size = inputs.shape[2]
            # img_grid = torchvision.utils.make_grid(inputs[:, :, cube_size // 2])
            # writer.add_image("train_images", img_grid)
            # writer.add_graph(model, inputs)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            probs = nn.functional.softmax(outputs, 1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # print batch result
            lr = [group['lr'] for group in optimizer.param_groups]
            if itr % print_per_iteration == 0:
                e_time = time.time()
                t = e_time - s_time
                acc = torch.sum(preds == labels.data).double() / len(preds)
                print("{:}: EPOCH{:03d} {:}Itr{:}/{:} ({:.2f}s/itr) Train: acc {:3.2f}, loss {:2.4f}, lr {:s}".format(
                    datetime.now(), epoch, int(print_per_iteration), itr // print_per_iteration,
                    len(trainLoader) // print_per_iteration, t, acc.item(), loss.item(), str(lr)))

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            itr += 1

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

        print("=" * 50)
        
        epoch_loss = running_loss / len(trainLoader.dataset)
        epoch_acc = running_corrects.double() / len(trainLoader.dataset)
        

        all_preds = np.concatenate(all_preds).reshape(-1)
        all_labels = np.concatenate(all_labels).reshape(-1)
        confMat = confusion_matrix(all_labels, all_preds, np.arange(num_classes))
        df_cm = pd.DataFrame(confMat, index=[i for i in range(num_classes)],
                             columns=[i for i in range(num_classes)])
        print("Train confusion matrix: ")
        print(df_cm)
        fig = plt.figure()
        sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
        plt.savefig(os.path.join(plot_folder, "train_confusion_matrix_ep{:d}.png".format(epoch)), bbox_inches="tight", dpi=200)
        plt.close(fig)

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        print("epoch {:d} | avg training loss {:.6f} | avg training acc {:.2f}".format(
              epoch, epoch_loss, epoch_acc))

        if testLoader:
            model.eval()

            scores = []
            correct = 0
            total = 0
            all_preds = []
            all_probs = []
            all_labels = []
            for sample_batch in testLoader:
                # inputs, labels = sample_batch["cubes"], sample_batch["label"]
                inputs, labels = sample_batch
                if isinstance(inputs, list):
                    # inputs = [x.float().to(device) for x in inputs]
                    inputs = torch.from_numpy(np.array(inputs).astype(np.float32)).to(device)
                    labels = torch.from_numpy(np.array(labels).astype(np.int)).to(device)
                else:
                    inputs = inputs.float().to(device)
                    labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                preds = torch.argmax(outputs, 1).cpu()
                probs = nn.functional.softmax(outputs, 1)[:, 1].cpu()
                labels = labels.cpu()
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                scores.append(loss.cpu().detach().numpy())
                all_probs.append(probs.detach().numpy())
                all_preds.append(preds.numpy())
                all_labels.append(labels.numpy())

            score = np.mean(scores)
            acc = correct / total * 100
            scheduler.step(score)

            all_preds = np.concatenate(all_preds).reshape(-1)
            all_labels = np.concatenate(all_labels).reshape(-1)
            confMat = confusion_matrix(all_labels, all_preds, np.arange(num_classes))
            df_cm = pd.DataFrame(confMat, index=[i for i in range(num_classes)],
                                 columns=[i for i in range(num_classes)])
            print("Test confusion matrix: ")
            print(df_cm)
            plt.figure()
            sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
            plt.savefig(os.path.join(plot_folder, "test_confusion_matrix_ep{:d}.png".format(epoch)),
                        bbox_inches="tight", dpi=200)
            plt.close()

            # print("epoch {:d}, test loss {:.6f}".format(epoch, score))
            print("epoch {:d} | avg test loss {:.6f} | avg test acc {:.2f}".format(
                epoch, score, acc))

            writer.add_scalar('Loss/test', score, epoch)
            writer.add_scalar('Accuracy/test', acc, epoch)

            test_loss.append(score)
            test_acc.append(acc)
            # test_step.append(step)

            # if auc_score > best_auc_score or (auc_score == best_auc_score and val_accs > best_acc):
            if score < best_val_loss or (score == best_val_loss and acc > best_acc):
                save_path = os.path.join(save_dir, 'epoch_' + str(epoch) + '.pt')
                torch.save(model.state_dict(), save_path)
                best_val_loss = score
                best_acc = acc
                best_epoch = epoch + 1

        else:
            scheduler.step(score)
        print("=" * 50)

        # if epoch % 5 == 0:
        #     torch.save(model.state_dict(), os.path.join(save_dir, 'epoch_' + str(epoch) + '.pt'))

    # Retrain on train_val dataset
    train_full_loss = []
    train_full_acc = []
    for epoch in tqdm(range(start_epoch, best_epoch)):
        print('Epoch {}/{}'.format(epoch + 1, best_epoch))
        print('-' * 10)

        # Training phase
        model_copy.train()
        all_preds = []
        all_probs = []
        all_labels = []

        running_loss = 0.0
        running_corrects = 0
        itr = 0
        s_time = time.time()

        for sample_batch in trainValLoader:
            # inputs, labels = sample_batch["cubes"], sample_batch["label"]
            inputs, labels = sample_batch
            if isinstance(inputs, list):
                inputs = torch.from_numpy(np.array(inputs).astype(np.float32)).to(device)
                labels = torch.from_numpy(np.array(labels).astype(np.int)).to(device)
            else:
                inputs = inputs.float().to(device)
                labels = labels.to(device)

            # cube_size = inputs.shape[2]
            # img_grid = torchvision.utils.make_grid(inputs[:, :, cube_size // 2])
            # writer.add_image("train_images", img_grid)
            # writer.add_graph(model, inputs)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model_copy(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            probs = nn.functional.softmax(outputs, 1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # print batch result
            lr = [group['lr'] for group in optimizer.param_groups]
            if itr % print_per_iteration == 0:
                e_time = time.time()
                t = e_time - s_time
                acc = torch.sum(preds == labels.data).double() / len(preds)
                print("TrainVal: {:}: EPOCH{:03d} {:}Itr{:}/{:} ({:.2f}s/itr) Train: acc {:3.2f}, loss {:2.4f}, lr {:s}".format(
                    datetime.now(), epoch, int(print_per_iteration), itr // print_per_iteration,
                                                                     len(trainLoader) // print_per_iteration, t,
                    acc.item(), loss.item(), str(lr)))

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            itr += 1

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

        print("=" * 50)

        epoch_loss = running_loss / len(trainValLoader.dataset)
        epoch_acc = running_corrects.double() / len(trainValLoader.dataset)

        all_preds = np.concatenate(all_preds).reshape(-1)
        all_labels = np.concatenate(all_labels).reshape(-1)
        confMat = confusion_matrix(all_labels, all_preds, np.arange(num_classes))
        df_cm = pd.DataFrame(confMat, index=[i for i in range(num_classes)],
                             columns=[i for i in range(num_classes)])
        print("Train confusion matrix: ")
        print(df_cm)
        fig = plt.figure()
        sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
        plt.savefig(os.path.join(plot_folder, "train_full_confusion_matrix_ep{:d}.png".format(epoch)), bbox_inches="tight",
                    dpi=200)
        plt.close(fig)

        writer.add_scalar('Loss/train_full', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train_full', epoch_acc, epoch)

        train_full_loss.append(epoch_loss)
        train_full_acc.append(epoch_acc)

        print("epoch {:d} | avg training loss {:.6f} | avg training acc {:.2f}".format(
            epoch, epoch_loss, epoch_acc))

    torch.save(model_copy.state_dict(), os.path.join(save_dir, 'full_epoch_' + str(best_epoch) + '.pt'))

    if testLoader:
        # start_from_step = 200
        train_loss = np.array(train_loss)
        # plt.plot(np.arange(start_from_step, len(train_loss)), train_loss[start_from_step:], label="Train loss")
        # ids = np.array(test_step) > start_from_step
        # plt.plot(np.array(test_step)[ids], np.array(test_loss)[ids], label="Test loss")

        plt.figure()
        plt.plot(train_loss, label="Train loss")
        plt.plot(test_loss, label="Test loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        # plt.yscale("log")
        plot_name = "loss curve_ep{:d}.png".format(epochs)
        plt.savefig(os.path.join(save_dir, plot_name), dpi=200, bbox_inches="tight")
        # plt.show()
        plt.close()

        plt.figure()
        plt.plot(train_acc, label="Train acc")
        plt.plot(test_acc, label="Test acc")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        # plt.yscale("log")
        plot_name = "acc curve_ep{:d}.png".format(epochs)
        plt.savefig(os.path.join(save_dir, plot_name), dpi=200, bbox_inches="tight")
        # plt.show()
        plt.close()

    return model_copy, best_epoch

def test(testLoader, model, criterion, save_folder, start_epoch):

    # epoch = 20
    # model.load_state_dict(torch.load(os.path.join(save_dir, 'epoch_' + str(epoch) + '.pt')))
    # print("load model from: {:s}".format(os.path.join(save_dir, 'epoch_' + str(epoch) + '.pt')))
    model.eval()

    num_classes = 2
    scores = []
    correct = 0
    total = 0
    all_preds = []
    all_probs = []
    all_labels = []
    for sample_batch in tqdm(testLoader):
        # inputs, labels = sample_batch["cubes"], sample_batch["label"]
        inputs, labels = sample_batch
        if isinstance(inputs, list):
            # inputs = [x.float().to(device) for x in inputs]
            inputs = torch.from_numpy(np.array(inputs).astype(np.float32)).to(device)
            labels = torch.from_numpy(np.array(labels).astype(np.int)).to(device)
        else:
            inputs = inputs.float().to(device)
            labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        preds = torch.argmax(outputs, 1).cpu()
        probs = nn.functional.softmax(outputs, 1)[:, 1].cpu()
        labels = labels.cpu()
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        scores.append(loss.cpu().detach().numpy())
        all_probs.append(probs.detach().numpy())
        all_preds.append(preds.numpy())
        all_labels.append(labels.numpy())

    score = np.mean(scores)
    acc = correct / total * 100

    all_preds = np.concatenate(all_preds).reshape(-1)
    all_labels = np.concatenate(all_labels).reshape(-1)

    np.savez_compressed(os.path.join(save_folder, "preds.npz"),
                        l=all_labels, p=all_probs)

    confMat = confusion_matrix(all_labels, all_preds, np.arange(num_classes))
    df_cm = pd.DataFrame(confMat, index=[i for i in range(num_classes)],
                         columns=[i for i in range(num_classes)])
    print("Test confusion matrix: ")
    print(df_cm)
    plt.figure()
    sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
    plot_folder = os.path.join(save_folder, "test")
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder, "test_confusion_matrix_ep{:d}.png".format(start_epoch)),
                bbox_inches="tight", dpi=200)
    plt.close()

    # print("epoch {:d}, test loss {:.6f}".format(epoch, score))
    print("epoch {:d} | avg test loss {:.6f} | avg test acc {:.2f}".format(
        start_epoch, score, acc))


    # auc_score = roc_auc_score(labels_test[:, 0], probs_test[:, 0])
    # print("test loss {:.2f} | test acc {:.4f} | auc score {:.4f}".format(
    #     loss_test, acc_test, auc_score))
    #
    # all_labels = np.argmax(labels_test, axis=-1)
    # all_preds = np.argmax(probs_test, axis=-1)
    #
    #
    # fpr, tpr, ths = roc_curve(labels_test[:, 0], probs_test[:, 0])
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.savefig(os.path.join(model_dir, "roc_curve.png"), bbox_inches="tight", dpi=200)
    # plt.close()
    #
    # # optimal_idx = np.argmax(tpr - fpr)
    # # optimal_threshold = ths[optimal_idx]
    #
    # confMat = confusion_matrix(all_labels, all_preds)
    # df_cm = pd.DataFrame(confMat, index=["maligant", "benign"],
    #                      columns=["maligant", "benign"])
    # print("Test confusion matrix with th_0.5:")
    # print(df_cm)
    # plt.figure()
    # # sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
    # sns.heatmap(df_cm, annot=True, cmap="Blues")
    # plt.xlabel("Predicted label")
    # plt.ylabel("True label")
    # plt.title("Confusion matrix")
    # plt.savefig(os.path.join(model_dir, "test_confusion_matrix_th_0.5.png"), bbox_inches="tight", dpi=200)
    # plt.close()
    #
    #
    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = ths[optimal_idx]
    # all_pred_new = 1 - (probs_test[:, 0] >= optimal_threshold).astype(np.int)
    # confMat = confusion_matrix(all_labels, all_pred_new)
    # df_cm = pd.DataFrame(confMat, index=["maligant", "benign"],
    #                      columns=["maligant", "benign"])
    # from sklearn.metrics import classification_report
    # print("Classification report with th_{:f}: ".format(optimal_threshold))
    # print(classification_report(all_labels, all_pred_new))
    # print("Test confusion matrix with th_{:f}: ".format(optimal_threshold))
    # print(df_cm)
    # plt.figure()
    # # sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
    # sns.heatmap(df_cm, annot=True, cmap="Blues")
    # plt.xlabel("Predicted label")
    # plt.ylabel("True label")
    # plt.title("Confusion matrix")
    # plt.savefig(os.path.join(model_dir, "test_confusion_matrix_th_{:.3f}.png".format(optimal_threshold)),
    #             bbox_inches="tight", dpi=200)
    # plt.close()
    #
    #
    # select_th = ths[np.argmax(tpr >= 0.8)]
    # all_pred_new = 1 - (probs_test[:, 0] >= select_th).astype(np.int)
    # confMat = confusion_matrix(all_labels, all_pred_new)
    # df_cm = pd.DataFrame(confMat, index=["maligant", "benign"],
    #                      columns=["maligant", "benign"])
    # from sklearn.metrics import classification_report
    # print("Classification report with th_{:f}: ".format(select_th))
    # print(classification_report(all_labels, all_pred_new))
    # print("Test confusion matrix with th_{:f}: ".format(select_th))
    # print(df_cm)
    # plt.figure()
    # # sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
    # sns.heatmap(df_cm, annot=True, cmap="Blues")
    # plt.xlabel("Predicted label")
    # plt.ylabel("True label")
    # plt.title("Confusion matrix")
    # plt.savefig(os.path.join(model_dir, "test_confusion_matrix_th_{:.3f}.png".format(select_th)),
    #             bbox_inches="tight", dpi=200)
    # plt.close()
    #
    #
    # average_precision = average_precision_score(labels_test[:, 0], probs_test[:, 0])
    # precision, recall, thresholds = precision_recall_curve(labels_test[:, 0], probs_test[:, 0])
    # plt.figure()
    # plt.plot(recall, precision, label='precision-recall curve (AP = %0.2f)' % average_precision)
    # plt.xlim([0.0, 1.05])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-recall curve')
    # plt.legend(loc="lower left")
    # plt.savefig(os.path.join(model_dir, "precision_recall_curve.png"), bbox_inches="tight", dpi=200)
    # plt.close()


def main():
    ## ----- Load parameters ----- ##
    datasource = args.datasource
    save_dir = args.save_dir
    load_model = args.load_model
    train_flag = args.train
    kfold = args.kfold
    splitId = args.splitId
    extra_str = args.extra_str
    model_name = args.model
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    ## ----- Create logger ----- ##
    model_folder = "{:s}_{:s}".format(model_name, extra_str) if len(extra_str) > 0 else model_name
    if args.kfold: model_folder += ".kfold{:d}".format(splitId)
    save_dir = os.path.join(save_dir, model_folder)
    os.makedirs(save_dir, exist_ok=True)
    log_file = "train.log" if train_flag else "test.log"
    log_path = os.path.join(save_dir, log_file)
    sys.stdout = Logger(log_path)

    bind = lambda x: "--{:s}={:s}".format(str(x[0]), str(x[1]))
    print("=" * 100)
    print("Running at: {:s}".format(str(datetime.now())))
    print("Working in directory: {:s}\n".format(save_dir))
    print("Run experiments: ")
    print("python {:s}".format(" ".join(sys.argv)))
    print("Full arguments: ")
    print("{:s}\n".format(" ".join([bind(i) for i in vars(args).items()])))

    global writer
    writer = SummaryWriter(os.path.join(save_dir, "run"))


    ## ----- Create datasets and dataLoaders ----- ##
    if datasource == "methodist":
        from dataLoader.IncidentalData import LungDataset, IncidentalConfig
        from utils.model_utils import collate

        config = IncidentalConfig()
        lungData = LungDataset(config)
        kfold = KFold(n_splits=args.kfold, random_state=42) if kfold is not None else None
        datasets = lungData.get_datasets(kfold=kfold, splitId=splitId)
        # kfold = len(lungData.y) if kfold is None else args.kfold
        # kfold = KFold(n_splits=kfold, random_state=42)

        # trainLoader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate)
        trainLoader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
        trainValLoader = DataLoader(datasets["train_val"], batch_size=batch_size, shuffle=True)
        valLoader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
        testLoader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)

        # trainData = LungDataset(config, "train", kfold=kfold, splitId=splitId)
        # trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True, collate_fn=collate)
        # valData = LungDataset(config, "val", kfold=kfold, splitId=splitId)
        # valLoader = DataLoader(valData, batch_size=batch_size, shuffle=False)


        # pos_label_file = "/data/pyuan2/Methodist_incidental/data_Ben/labeled/pos_labels_norm.csv"
        # cat_label_file = "../data/Lung Nodule Clinical Data_Min Kim - Added Variables 10-2-2020.xlsx"
        # load_model = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/model/classification_LUNA16/Resnet18_Adam_lr0.001"
        # load_model = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/model/kim_labeled_198/Resnet18_"

        # trainData = LungDataset(root_dir, pos_label_file=args.pos_label_path, cat_label_file=args.cat_label_path,
        #                        cube_size=cube_size, train=True, screen=True, clinical=clinical)
        # trainData = LUNA16(train=True)


        # valData = LungDataset(root_dir, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
        #                       cube_size=cube_size, train=False, screen=True, clinical=clinical)
        # valData = LUNA16(train=False)
        # valLoader = DataLoader(valData, batch_size=1, shuffle=False)
    else:
        assert datasource == "luna"
        from dataLoader.LUNA16Data import LUNA16, LunaConfig
        config = LunaConfig()
        # root_dir = "../data/"
        # pos_label_file = "../data/pos_labels.csv"
        # cat_label_file = "../data/Lung Nodule Clinical Data_Min Kim (No name).xlsx"
        # load_model = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/model/classification_LUNA16/Resnet18_Adam_lr0.001"
        # cube_size = 48
        trainData = LUNA16(train=True)
        trainLoader = DataLoader(trainData, batch_size=2, shuffle=True)
        valData = LUNA16(train=False)
        valLoader = DataLoader(valData, batch_size=1, shuffle=False)

    print("Shape of train_x is: ", (len(datasets["train"]), 1,) + (config.CUBE_SIZE,) * 3)
    print("Shape of train_y is: ", (len(datasets["train"]),))
    print("Shape of val_x is: ", (len(datasets["val"]), 1,) + (config.CUBE_SIZE,) * 3)
    print("Shape of val_y is: ", (len(datasets["val"]),))
    config.display()

    ## ----- Construct models ----- ##

    # model_name = "Resnet18"
    # extra_str = "SGD_lr0.001"
    # extra_str = "Adam_lr0.001_augment"
    # extra_str = "Adam_lr0.001"
    # extra_str = "Test_for_incidental_48_all"
    # extra_str = ""
    # if clinical:
    #     extra_str += "additional_clinical"
    if model_name == "res18":
        model = generate_model(18, n_input_channels=1, n_classes=2, clinical=config.LOAD_CLINICAL)
    print("Use model: {:s}".format(model_name))
    # save_dir = "model/classification_negMultiple/"
    # save_dir = "model/classification_LUNA16/"
    # save_dir = "model/classification_169patients/"
    # save_dir = "model/kim_labeled_169/"
    # if datasource == "methodist":
    #     save_dir = "model/kim_labeled_198/"
    # elif datasource == "luna":
    #     save_dir = "model/classification_LUNA16/"
    # save_dir += "{:s}_{:s}".format(model_name, extra_str)



    if load_model:
        model_list = [m for m in os.listdir(load_model) if m.endswith("pt")]
        from natsort import natsorted
        latest_model = natsorted(model_list)[-1]
        start_epoch = int(latest_model.strip(".pt")[6:])
        model_path = os.path.join(load_model, latest_model)
        state_dict = torch.load(model_path)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = "module." + k  # add "module." for dataparallel
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        print("Load successfully from " + model_path)
    else:
        start_epoch = 0
        
    # Print the model we just instantiated
    print(model)

    # Detect if we have a GPU available
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     print("using GPU now!")

    # Send the model to GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("Using {:} GPUs".format(torch.cuda.device_count()))
    model = model.to(device)
    
    ## ----- Set criterion, optimizer ----- ##
    # criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # criterion = RMSLELoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=10, min_lr=0.0001, patience=100)

    # scaler = RobustScaler()
    # train_x = scaler.fit_transform(train_x)
    # test_x = scaler.transform(test_x)
    # transform_x = lambda x: torch.from_numpy(x).float()
    # if featureFile in ["features_v2", "features_kaggle"]:
    #     scale_y_func = lambda x: np.array(x.reshape(-1))
    # else:
    #     scale_y_func = lambda x: np.log1p(x.reshape(-1))
    # transform_y = transforms.Compose([
    #     scale_y_func,
    #     lambda x: torch.from_numpy(x).float()
    # ])

    # epoch = 20
    # model.load_state_dict(torch.load(os.path.join(load_model, 'epoch_' + str(epoch) + '.pt')))
    # print("load model from: {:s}".format(os.path.join(load_model, 'epoch_' + str(epoch) + '.pt')))


    # Create dataLoader for training the model
    # pseudo_train_x, pseudo_test_x, pseudo_train_y, pseudo_test_y = train_test_split(
    #     train_x, train_y, test_size=0.2, random_state=42)
    # trainData = HouseData(pseudo_train_x, pseudo_train_y,
    #                       transform=transform_x,
    #                       target_transform=transform_y)
    # trainLoader = DataLoader(trainData, batch_size=100, shuffle=True)
    # testData = HouseData(pseudo_test_x, pseudo_test_y,
    #                      transform=transform_x,
    #                      target_transform=transform_y)
    # testLoader = DataLoader(testData, batch_size=len(pseudo_test_x), shuffle=False)
    # Train the model
    
    ## ----- Training or test ----- ##
    if train_flag:
        model, start_epoch = train(trainLoader, valLoader, trainValLoader, model, optimizer, scheduler, criterion, save_dir, epochs, start_epoch)
        test(testLoader, model, criterion, save_dir, start_epoch)
    else:
        test(testLoader, model, criterion, save_dir, start_epoch)


    # # Create dataLoader for the final test data
    # finalTestData = HouseData(test_x, np.zeros([len(test_x)]),
    #                           transform=transform_x,
    #                           target_transform=transform_y)
    # finalTestLoader = DataLoader(finalTestData, batch_size=len(test_x), shuffle=False)
    # # Test the model (Run prediction)
    # test(finalTestLoader, model, device)



if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Spent {:.2f}s".format(time.time() - start_time))
