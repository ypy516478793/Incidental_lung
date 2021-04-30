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

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torchvision
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

def train(trainLoader, testLoader, model, optimizer, scheduler, criterion, device, model_folder, start_epoch=0):
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
    for epoch in tqdm(range(start_epoch, epochs)):
        model.train()
        scores = []
        correct = 0
        total = 0
        all_pred = []
        all_label = []
        for sample_batch in trainLoader:
            x1, y1 = sample_batch["cubes"], sample_batch["label"]
            if isinstance(x1, list):
                x1 = torch.from_numpy(np.array(x1).astype(np.float32)).to(device)
                y1 = torch.from_numpy(np.array(y1).astype(np.int)).to(device)
                # x1 = [x.float().to(device) for x in x1]
            else:
                x1 = x1.float().to(device)
                y1 = y1.to(device)

            # cube_size = x1.shape[2]
            # img_grid = torchvision.utils.make_grid(x1[:, :, cube_size // 2])
            # writer.add_image("train_images", img_grid)
            # writer.add_graph(model, x1)

            optimizer.zero_grad()
            p1 = model(x1)
            loss = criterion(p1, y1)
            loss.backward()

            score = loss.cpu().detach().numpy()
            scores.append(score)
            lr = [group['lr'] for group in optimizer.param_groups]
            print("epoch {:d}, step {:d}, training loss: {:.6f}, learning rate: {:s}".format(
                epoch, step, score, str(lr)))
            # train_loss.append(score)
            optimizer.step()

            c1 = torch.argmax(p1, 1).cpu()
            y1 = y1.cpu()
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
        fig = plt.figure()
        sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
        plt.savefig(os.path.join(plot_folder, "train_confusion_matrix_ep{:d}.png".format(epoch)), bbox_inches="tight", dpi=200)
        plt.close()

        writer.add_scalar('Loss/train', score, epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)

        train_loss.append(score)
        train_acc.append(acc)

        print("epoch {:d} | avg training loss {:.6f} | avg training acc {:.2f}".format(
              epoch, score, acc))

        if testLoader:
            model.eval()
            # sample_batch = next(iter(testLoader))
            # x2, y2 = sample_batch["cubes"].float(), sample_batch["label"]
            # p2 = model(x2.to(device))
            # loss = criterion(p2, y2.to(device))
            # score = loss.cpu().detach().numpy()

            scores = []
            correct = 0
            total = 0
            all_pred = []
            all_label = []
            for sample_batch in testLoader:
                x2, y2 = sample_batch["cubes"], sample_batch["label"]
                if isinstance(x2, list):
                    # x2 = [x.float().to(device) for x in x2]
                    x2 = torch.from_numpy(np.array(x2).astype(np.float32)).to(device)
                    y2 = torch.from_numpy(np.array(y2).astype(np.int)).to(device)
                else:
                    x2 = x2.float().to(device)
                    y2 = y2.to(device)
                optimizer.zero_grad()
                p2 = model(x2)
                loss = criterion(p2, y2)
                loss.backward()

                c2 = torch.argmax(p2, 1).cpu()
                y2 = y2.cpu()
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

            writer.add_scalar('Loss/test', score, epoch)
            writer.add_scalar('Accuracy/test', acc, epoch)

            test_loss.append(score)
            test_acc.append(acc)
            test_step.append(step)
        else:
            scheduler.step(score)
        print("=" * 50)

        if epoch % 5 == 0:
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

def test(testLoader, model, device, criterion, save_folder):

    # epoch = 20
    # model.load_state_dict(torch.load(os.path.join(model_folder, 'epoch_' + str(epoch) + '.pt')))
    # print("load model from: {:s}".format(os.path.join(model_folder, 'epoch_' + str(epoch) + '.pt')))
    model.eval()

    num_classes = 2
    scores = []
    correct = 0
    total = 0
    all_pred = []
    all_label = []
    for sample_batch in tqdm(testLoader):
        x2, y2 = sample_batch["cubes"], sample_batch["label"]
        if isinstance(x2, list):
            # x2 = [x.float().to(device) for x in x2]
            x2 = torch.from_numpy(np.array(x2).astype(np.float32)).to(device)
            y2 = torch.from_numpy(np.array(y2).astype(np.int)).to(device)
        else:
            x2 = x2.float().to(device)
            y2 = y2.to(device)
        p2 = model(x2)
        loss = criterion(p2, y2)
        loss.backward()

        c2 = torch.argmax(p2, 1).cpu()
        y2 = y2.cpu()
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


    # auc_score = roc_auc_score(labels_test[:, 0], probs_test[:, 0])
    # print("test loss {:.2f} | test acc {:.4f} | auc score {:.4f}".format(
    #     loss_test, acc_test, auc_score))
    #
    # all_label = np.argmax(labels_test, axis=-1)
    # all_pred = np.argmax(probs_test, axis=-1)
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
    # confMat = confusion_matrix(all_label, all_pred)
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
    # confMat = confusion_matrix(all_label, all_pred_new)
    # df_cm = pd.DataFrame(confMat, index=["maligant", "benign"],
    #                      columns=["maligant", "benign"])
    # from sklearn.metrics import classification_report
    # print("Classification report with th_{:f}: ".format(optimal_threshold))
    # print(classification_report(all_label, all_pred_new))
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
    # confMat = confusion_matrix(all_label, all_pred_new)
    # df_cm = pd.DataFrame(confMat, index=["maligant", "benign"],
    #                      columns=["maligant", "benign"])
    # from sklearn.metrics import classification_report
    # print("Classification report with th_{:f}: ".format(select_th))
    # print(classification_report(all_label, all_pred_new))
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

    Train = True
    use_clinical_features = False
    # rootFolder = "../data_king/labeled"
    rootFolder = "/data/pyuan2/Methodist_incidental/data_Ben/labeled/"
    pos_label_file = "/data/pyuan2/Methodist_incidental/data_Ben/labeled/pos_labels_norm.csv"
    cat_label_file = "../data/Lung Nodule Clinical Data_Min Kim - Added Variables 10-2-2020.xlsx"
    # load_model_folder = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/model/classification_LUNA16/Resnet18_Adam_lr0.001"
    load_model_folder = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/model/kim_labeled_198/Resnet18_"
    cube_size = 64
    trainData = LungDataset(rootFolder, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
                           cube_size=cube_size, train=True, screen=True, clinical=use_clinical_features)
    # trainData = LUNA16(train=True)
    from utils import collate
    trainLoader = DataLoader(trainData, batch_size=3, shuffle=True, collate_fn=collate)

    valData = LungDataset(rootFolder, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
                          cube_size=cube_size, train=False, screen=True, clinical=use_clinical_features)
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
    extra_str = ""
    # extra_str = "SGD_lr0.001"
    # extra_str = "Adam_lr0.001_augment"
    # extra_str = "Adam_lr0.001"
    # extra_str = "Test_for_incidental_48_all"
    # extra_str = ""
    if use_clinical_features:
        extra_str += "additional_clinical"
    model = generate_model(18, n_input_channels=1, n_classes=2, clinical=use_clinical_features)
    print("Use model: {:s}".format(modelName))
    # model_folder = "model/classification_negMultiple/"
    # model_folder = "model/classification_LUNA16/"
    # model_folder = "model/classification_169patients/"
    # model_folder = "model/kim_labeled_169/"
    model_folder = "model/kim_labeled_198/"
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

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=10, min_lr=0.0001, patience=100)

    global writer
    writer = SummaryWriter(os.path.join(model_folder, "run"))

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
    # model.load_state_dict(torch.load(os.path.join(load_model_folder, 'epoch_' + str(epoch) + '.pt')))
    # print("load model from: {:s}".format(os.path.join(load_model_folder, 'epoch_' + str(epoch) + '.pt')))

    if load_model_folder:
        model_list = [m for m in os.listdir(load_model_folder) if m.endswith("pt")]
        from natsort import natsorted
        latest_model = natsorted(model_list)[-1]
        start_epoch = int(latest_model.strip(".pt")[6:])
        model_path = os.path.join(load_model_folder, latest_model)
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
        train(trainLoader, valLoader, model, optimizer, scheduler, criterion, device, model_folder, start_epoch)
    else:
        test(valLoader, model, device, criterion, model_folder)


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
