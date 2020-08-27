from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from IncidentalData import LungDataset
# from kaggleData import HouseData
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from classifier.resnet import generate_model

import torch.optim as optim
import torch.nn as nn
import torch

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

def train(trainLoader, testLoader, model, optimizer, scheduler, criterion, device):
    epochs = 500
    step = 0
    train_loss = []
    test_loss = []
    test_step = []
    for epoch in tqdm(range(epochs)):
        model.train()
        for sample_batch in trainLoader:
            x1, y1 = sample_batch["features"].float(), sample_batch["label"]
            optimizer.zero_grad()
            p1 = model(x1.to(device))
            loss = criterion(p1, y1.to(device))
            loss.backward()

            score = loss.cpu().detach().numpy()
            lr = [group['lr'] for group in optimizer.param_groups]
            print("epoch {:d}, step {:d}, training loss: {:.4f}, learning rate: {:s}".format(
                epoch, step, score, str(lr)))
            train_loss.append(score)
            optimizer.step()
            step += 1

        if testLoader:
            model.eval()
            sample_batch = next(iter(testLoader))
            x2, y2 = sample_batch["features"].float(), sample_batch["label"]
            p2 = model(x2.to(device))
            loss = criterion(p2, y2.to(device))
            score = loss.cpu().detach().numpy()
            scheduler.step(score)
            print("epoch {:d}, test loss {:.4f}".format(epoch, score))
            test_loss.append(score)
            test_step.append(step)
        else:
            scheduler.step(score)

    if testLoader:
        import matplotlib.pyplot as plt
        start_from_step = 200
        train_loss = np.array(train_loss)
        plt.plot(np.arange(start_from_step, len(train_loss)), train_loss[start_from_step:], label="Train loss")
        ids = np.array(test_step) > start_from_step
        plt.plot(np.array(test_step)[ids], np.array(test_loss)[ids], label="Test loss")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("loss")
        # plt.yscale("log")
        plt.savefig("loss curve.png", dpi=200, bbox_inches="tight")
        plt.show()

    return model

def test(finalTestLoader, model, device):
    sample_batch = next(iter(finalTestLoader))
    x, y = sample_batch["features"], sample_batch["label"]
    pred = model(x.to(device))
    test_y_hat = torch.expm1(pred).cpu().detach().numpy()

    test_ID = np.arange(1461, 2920).astype(np.int)
    sub = pd.DataFrame()
    sub["Id"] = test_ID
    sub["SalePrice"] = test_y_hat
    sub.to_csv("submission.csv", index=False)
    print("Save predictions to: {:s}".format("submission.csv"))


def main():

    rootFolder = "../data/"
    pos_label_file = "../data/pos_labels.csv"
    cat_label_file = "../data/Lung Nodule Clinical Data_Min Kim (No name).xlsx"
    cube_size = 64
    trainData = LungDataset(rootFolder, labeled_only=True, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
                            cube_size=cube_size, reload=False, train=True)
    trainLoader = DataLoader(trainData, batch_size=2, shuffle=True)

    valData = LungDataset(rootFolder, labeled_only=True, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
                          cube_size=cube_size, reload=False, train=False)
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
    model = generate_model(18, n_input_channels=1, n_classes=2)
    print("Use model: {:s}".format(modelName))

    start_time = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    # criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # criterion = RMSLELoss()
    criterion = nn.CrossEntropyLoss()
    # criterion_test = nn.L1Loss()

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=10, min_lr=0.0001, patience=50)

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
    model = train(trainLoader, valLoader, model, optimizer, scheduler, criterion, device)


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
