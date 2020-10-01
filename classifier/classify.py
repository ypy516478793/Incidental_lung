import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from IncidentalData import LungDataset
from tqdm import tqdm

from classifier.resnet import generate_model
from classifier.mlp import Net

import torch.optim as optim
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import os

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def train(trainLoader, testLoader, model, optimizer, scheduler, criterion, device, model_folder):
    epochs = 51
    step = 0
    num_classes = 2
    plot_folder = os.path.join(model_folder, "plots")
    os.makedirs(plot_folder, exist_ok=True)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    test_step = []
    for epoch in tqdm(range(0, epochs)):
        model.train()
        scores = []
        correct = 0
        total = 0
        all_pred = []
        all_label = []
        for sample_batch in trainLoader:
            x1, y1 = sample_batch["features"].float(), sample_batch["label"]
            optimizer.zero_grad()
            p1 = model(x1.to(device))
            loss = criterion(p1, y1.to(device))
            loss.backward()

            score = loss.cpu().detach().numpy()
            scores.append(score)
            lr = [group['lr'] for group in optimizer.param_groups]
            print("epoch {:d}, step {:d}, training loss: {:.6f}, learning rate: {:s}".format(
                epoch, step, score, str(lr)))
            optimizer.step()

            c1 = torch.argmax(p1, 1).cpu()
            total += y1.size(0)
            correct += (c1 == y1).sum().item()
            all_pred.append(c1.numpy())
            all_label.append(y1.numpy())

            step += 1

        print("=" * 50)

        score = np.mean(scores)
        acc = correct / total * 100
        all_pred = np.concatenate(all_pred).reshape(-1)
        all_label = np.concatenate(all_label).reshape(-1)
        confMat = confusion_matrix(all_label, all_pred)
        df_cm = pd.DataFrame(confMat, index=[i for i in range(num_classes)],
                             columns=[i for i in range(num_classes)])
        print("Train confusion matrix: ")
        print(df_cm)
        plt.figure()
        sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
        plt.savefig(os.path.join(plot_folder, "train_confusion_matrix_ep{:d}.png".format(epoch)), bbox_inches="tight", dpi=200)
        plt.close()

        train_loss.append(score)
        train_acc.append(acc)

        print("epoch {:d} | avg training loss {:.6f} | avg training acc {:.2f}".format(
              epoch, score, acc))

        if testLoader:
            model.eval()

            scores = []
            correct = 0
            total = 0
            all_pred = []
            all_label = []
            for sample_batch in testLoader:
                x2, y2 = sample_batch["features"].float(), sample_batch["label"]
                optimizer.zero_grad()
                p2 = model(x2.to(device))
                loss = criterion(p2, y2.to(device))
                loss.backward()

                c2 = torch.argmax(p2, 1).cpu()
                total += y2.size(0)
                correct += (c2 == y2).sum().item()
                scores.append(loss.cpu().detach().numpy())
                all_pred.append(c2.numpy())
                all_label.append(y2.numpy())

            score = np.mean(scores)
            acc = correct / total * 100
            scheduler.step(score)

            all_pred = np.array(all_pred).reshape(-1)
            all_label = np.array(all_label).reshape(-1)
            confMat = confusion_matrix(all_label, all_pred)
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
            test_loss.append(score)
            test_acc.append(acc)
            test_step.append(step)
        else:
            scheduler.step(score)
        print("=" * 50)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_folder, 'epoch_' + str(epoch) + '.pt'))

    if testLoader:
        train_loss = np.array(train_loss)
        plt.plot(train_loss, label="Train loss")
        plt.plot(test_loss, label="Test loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plot_name = "loss curve_ep{:d}.png".format(epochs)
        plt.savefig(os.path.join(model_folder, plot_name), dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()


        plt.plot(train_acc, label="Train acc")
        plt.plot(test_acc, label="Test acc")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plot_name = "acc curve_ep{:d}.png".format(epochs)
        plt.savefig(os.path.join(model_folder, plot_name), dpi=200, bbox_inches="tight")
        plt.show()

def test(testLoader, model, device, criterion, model_folder, save_folder):

    epoch = 50
    model.load_state_dict(torch.load(os.path.join(model_folder, 'epoch_' + str(epoch) + '.pt')))
    print("load model from: {:s}".format(os.path.join(model_folder, 'epoch_' + str(epoch) + '.pt')))
    model.eval()

    num_classes = 2
    scores = []
    correct = 0
    total = 0
    all_pred = []
    all_label = []
    for sample_batch in testLoader:
        x2, y2 = sample_batch["features"].float(), sample_batch["label"]
        p2 = model(x2.to(device))
        loss = criterion(p2, y2.to(device))
        loss.backward()

        c2 = torch.argmax(p2, 1).cpu()
        total += y2.size(0)
        correct += (c2 == y2).sum().item()
        scores.append(loss.cpu().detach().numpy())
        all_pred.append(c2.numpy())
        all_label.append(y2.numpy())

    score = np.mean(scores)
    acc = correct / total * 100

    all_pred = np.array(all_pred).reshape(-1)
    all_label = np.array(all_label).reshape(-1)
    confMat = confusion_matrix(all_label, all_pred)
    df_cm = pd.DataFrame(confMat, index=[i for i in range(num_classes)],
                         columns=[i for i in range(num_classes)])
    print("Test confusion matrix: ")
    print(df_cm)
    plt.figure()
    sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
    plot_folder = os.path.join(save_folder, "test")
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder, "test_confusion_matrix_ep{:d}.png".format(epoch)),
                bbox_inches="tight", dpi=200)
    plt.close()

    print("epoch {:d} | avg test loss {:.6f} | avg test acc {:.2f}".format(
        epoch, score, acc))


def main():

    Train = False
    use_clinical_features = False
    rootFolder = "../prepare_for_xinyue/"
    cat_label_file = "../prepare_for_xinyue/clinical_info.xlsx"
    load_model_folder = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/model/20patients/Resnet18_"
    cube_size = 64
    trainData = LungDataset(rootFolder, cat_label_file=cat_label_file, cube_size=cube_size, train=None, clinical=use_clinical_features)
    trainLoader = DataLoader(trainData, batch_size=2, shuffle=True)

    valData = LungDataset(rootFolder, cat_label_file=cat_label_file, cube_size=cube_size, train=None, clinical=use_clinical_features)
    valLoader = DataLoader(valData, batch_size=1, shuffle=False)

    print("Shape of train_x is: ", (len(trainData), 1,) + (cube_size,) * 3)
    print("Shape of train_y is: ", (len(trainData),))
    print("Shape of val_x is: ", (len(valData), 1,) + (cube_size,) * 3)
    print("Shape of val_y is: ", (len(valData),))

    extra_str = ""
    if not use_clinical_features:
        modelName = "Resnet18"
        model = generate_model(18, n_input_channels=1, n_classes=2)
    else:
        extra_str += "clinical"
        modelName = "Multilayer_perceptron"
        model = Net(26, output_dim=2)
    print("Use model: {:s}".format(modelName))
    model_folder = "model/20patients/"
    model_folder += "{:s}_{:s}".format(modelName, extra_str)
    os.makedirs(model_folder, exist_ok=True)

    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=10, min_lr=0.0001, patience=100)

    if Train:
        train(trainLoader, valLoader, model, optimizer, scheduler, criterion, device, model_folder)
    else:
        test(valLoader, model, device, criterion, load_model_folder, model_folder)

    print("Spent {:.2f}s".format(time.time() - start_time))

if __name__ == '__main__':
    main()
