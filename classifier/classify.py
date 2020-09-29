from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from IncidentalData import LungDataset
from LUNA16Data import LUNA16
from torchvision import transforms
from tqdm import tqdm

from classifier.resnet import generate_model

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
    epochs = 30
    step = 0
    num_classes = 2
    plot_folder = os.path.join(model_folder, "plots")
    os.makedirs(plot_folder, exist_ok=True)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    test_step = []
    for epoch in tqdm(range(epochs)):
        model.train()
        scores = []
        correct = 0
        total = 0
        all_pred = []
        all_label = []
        for sample_batch in trainLoader:
            x1, y1 = sample_batch["features"], sample_batch["label"]
            if isinstance(x1, list):
                x1 = [x.float().to(device) for x in x1]
            else:
                x1 = x1.float().to(device)
            optimizer.zero_grad()
            p1 = model(x1)
            loss = criterion(p1, y1.to(device))
            loss.backward()

            score = loss.cpu().detach().numpy()
            scores.append(score)
            lr = [group['lr'] for group in optimizer.param_groups]
            print("epoch {:d}, step {:d}, training loss: {:.6f}, learning rate: {:s}".format(
                epoch, step, score, str(lr)))
            # train_loss.append(score)
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
            # sample_batch = next(iter(testLoader))
            # x2, y2 = sample_batch["features"].float(), sample_batch["label"]
            # p2 = model(x2.to(device))
            # loss = criterion(p2, y2.to(device))
            # score = loss.cpu().detach().numpy()

            scores = []
            correct = 0
            total = 0
            all_pred = []
            all_label = []
            for sample_batch in testLoader:
                x2, y2 = sample_batch["features"], sample_batch["label"]
                if isinstance(x2, list):
                    x2 = [x.float().to(device) for x in x2]
                else:
                    x2 = x2.float().to(device)
                optimizer.zero_grad()
                p2 = model(x2)
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
        # start_from_step = 200
        train_loss = np.array(train_loss)
        # plt.plot(np.arange(start_from_step, len(train_loss)), train_loss[start_from_step:], label="Train loss")
        # ids = np.array(test_step) > start_from_step
        # plt.plot(np.array(test_step)[ids], np.array(test_loss)[ids], label="Test loss")

        plt.plot(train_loss, label="Train loss")
        plt.plot(test_loss, label="Test loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        # plt.yscale("log")
        plot_name = "loss curve_ep{:d}.png".format(epochs)
        plt.savefig(os.path.join(model_folder, plot_name), dpi=200, bbox_inches="tight")
        plt.show()


        plt.plot(train_acc, label="Train acc")
        plt.plot(test_acc, label="Test acc")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        # plt.yscale("log")
        plot_name = "acc curve_ep{:d}.png".format(epochs)
        plt.savefig(os.path.join(model_folder, plot_name), dpi=200, bbox_inches="tight")
        plt.show()

    # return model

def test(testLoader, model, device, criterion, model_folder, save_folder):

    epoch = 20
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
        x2, y2 = sample_batch["features"], sample_batch["label"]
        if isinstance(x2, list):
            x2 = [x.float().to(device) for x in x2]
        else:
            x2 = x2.float().to(device)
        p2 = model(x2)
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

    # print("epoch {:d}, test loss {:.6f}".format(epoch, score))
    print("epoch {:d} | avg test loss {:.6f} | avg test acc {:.2f}".format(
        epoch, score, acc))


def main():

    Train = True
    use_clinical_features = True
    rootFolder = "../data/"
    pos_label_file = "../data/pos_labels.csv"
    cat_label_file = "../data/Lung Nodule Clinical Data_Min Kim (No name).xlsx"
    load_model_folder = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/model/classification_LUNA16/Resnet18_Adam_lr0.001"
    cube_size = 64
    trainData = LungDataset(rootFolder, labeled_only=True, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
                            cube_size=cube_size, reload=False, train=True, clinical=use_clinical_features)
    # trainData = LUNA16(train=True)
    trainLoader = DataLoader(trainData, batch_size=3, shuffle=True)

    valData = LungDataset(rootFolder, labeled_only=True, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
                          cube_size=cube_size, reload=False, train=False, clinical=use_clinical_features)
    # valData = LUNA16(train=False)
    valLoader = DataLoader(valData, batch_size=1, shuffle=False)

    print("Shape of train_x is: ", (len(trainData), 1,) + (cube_size,) * 3)
    print("Shape of train_y is: ", (len(trainData),))
    print("Shape of val_x is: ", (len(valData), 1,) + (cube_size,) * 3)
    print("Shape of val_y is: ", (len(valData),))

    # featureFile = ["features_v1", "features_v2", "features_Sawyer", "features_kaggle"][1]
    # if featureFile == "features_kaggle":
    #     train_x = np.genfromtxt("processed_data/kaggle_solution_train_x.csv", delimiter=",")
    #     train_y = np.genfromtxt("processed_data/kaggle_solution_train_y.csv", delimiter=",")
    #     test_x = np.genfromtxt("processed_data/kaggle_solution_test_x.csv", delimiter=",")
    # else:
    #     file = os.path.join("./house-prices-advanced-regression-techniques", featureFile + ".csv")
    #     df_preprocessed = pd.read_csv(file, index_col = 'Id')
    #     num_train = 1458 if featureFile == "features_v2" else 1460
    #     train_x = df_preprocessed.iloc[:num_train, :-1].to_numpy()
    #     train_y = df_preprocessed.iloc[:num_train, -1].to_numpy()
    #     test_x = df_preprocessed.iloc[num_train:, :-1].to_numpy()
    #
    # print("Shape of train_x is: ", train_x.shape)
    # print("Shape of train_y is: ", train_y.shape)
    # print("Shape of test_x is: ", test_x.shape)

    modelName = "Resnet18"
    # extra_str = "SGD_lr0.001"
    # extra_str = "Adam_lr0.001_augment"
    # extra_str = "Adam_lr0.001"
    # extra_str = "Test_for_incidental_48_all"
    extra_str = ""
    if use_clinical_features:
        extra_str += "clinical"
    model = generate_model(18, n_input_channels=1, n_classes=2, clinical=use_clinical_features)
    print("Use model: {:s}".format(modelName))
    # model_folder = "model/classification_negMultiple/"
    # model_folder = "model/classification_LUNA16/"
    model_folder = "model/classification_169patients/"
    model_folder += "{:s}_{:s}".format(modelName, extra_str)
    os.makedirs(model_folder, exist_ok=True)

    start_time = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    # criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # criterion = RMSLELoss()
    criterion = nn.CrossEntropyLoss()
    # criterion_test = nn.L1Loss()

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    if Train:
        train(trainLoader, valLoader, model, optimizer, scheduler, criterion, device, model_folder)
    else:
        test(valLoader, model, device, criterion, load_model_folder, model_folder)


    # # Create dataLoader for the final test data
    # finalTestData = HouseData(test_x, np.zeros([len(test_x)]),
    #                           transform=transform_x,
    #                           target_transform=transform_y)
    # finalTestLoader = DataLoader(finalTestData, batch_size=len(test_x), shuffle=False)
    # # Test the model (Run prediction)
    # test(finalTestLoader, model, device)

    print("Spent {:.2f}s".format(time.time() - start_time))

if __name__ == '__main__':
    main()
