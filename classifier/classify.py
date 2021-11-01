"""
Instructions:
    python classify.py -d=luna --gpu=0,1,2,3 --save_dir=results/luna/ --train=True -b=16
"""

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from utils.plot_utils import plot_confusion_matrix
from utils.summary_utils import Logger
from utils.data_utils import augment
from datetime import datetime
from classifier.resnet import generate_model
# from classifier.resnet_rfs import ResNet_RFS
# from classifier.resnet_mb import ResNet_MB

import numpy as np
import time
import copy
import sys
import os

import argparse

parser = argparse.ArgumentParser(description="Incidental lung nodule classification")
parser.add_argument("-d", "--datasource", type=str, default="methodist", help="Dataset used for training/test",
                    choices=["luna", "lunaRaw", "methoidstPilot", "methodist", "additional"])
parser.add_argument("-p", "--data_dir", type=str, help="Data directory", default=None)
parser.add_argument("-s", "--save_dir", type=str, help="Save directory")
parser.add_argument("-g", "--gpu", type=str, default="0,1,2,3", help="Which gpus to use")

parser.add_argument("-m", "--model", default="res18", help="model")
parser.add_argument("-nc", "--n_classes", type=int, default=2, help="model")
parser.add_argument("-j", "--workers", default=0, type=int, help="number of data loading workers (default: 32)")
parser.add_argument("-lm", "--load_model", type=str, default=None, help="Path/Directory of the model to be loaded")
parser.add_argument("-t", "--train", type=eval, default=True, help="Train phase: True or False")
parser.add_argument("-re", "--resume", type=eval, default=False, help="Resume training")

parser.add_argument("-e", "--epochs", default=20, type=int, help="number of total epochs to run")
parser.add_argument("-b", "--batch_size", default=16, type=int, help="mini-batch size (default: 16)")
parser.add_argument("-op", "--optimizer", default="adam", type=str, help="adam or sgd")
parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, help="initial learning rate")
parser.add_argument("-mo", "--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("-wd", "--weight_decay", default=1e-3, type=float, help="weight dec)ay (default: 1e-4)")

parser.add_argument("-es", "--extra_str", type=str, default="", help="extra string for data")

parser.add_argument("-k", "--kfold", default=None, type=int, help="number of kfold for train_val")
parser.add_argument("-ki", "--splitId", default=None, type=int, help="split id when use kfold")
parser.add_argument("-ts", "--test_size", default=0.1, type=float, help="test size when use kfold")
parser.add_argument("-la", "--load_all", default=None, type=eval, help="whether to load all data")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
n_gpu = len(args.gpu.split(","))


from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def train(trainLoader, testLoader, trainValLoader, augmentor, model, optimizer, scheduler, criterion, save_dir,
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
    for epoch in tqdm(range(start_epoch, epochs)):
        print("Epoch {}/{}".format(epoch+1, epochs))
        print("-" * 10)

        # Training phase
        model.train()
        all_preds = []
        all_probs = []
        all_labels = []

        running_loss = 0.0
        running_corrects = 0
        s_time = time.time()

        for itr, sample_batch in enumerate(trainLoader):
            # Load inputs and labels
            inputs, labels = sample_batch
            inputs = augmentor(inputs)
            if isinstance(inputs, list) or isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(np.array(inputs).astype(np.float32)).to(device)
                labels = torch.from_numpy(np.array(labels).astype(np.int)).to(device)
            else:
                inputs = inputs.float().to(device)
                labels = labels.to(device)

            # Write image to tensorboard
            if itr < 2:
                cube_size = inputs.shape[2]
                img_grid = torchvision.utils.make_grid(inputs[:, :, cube_size // 2])
                writer.add_image("train_images_itr{:d}".format(itr), img_grid, global_step=epoch)
                # writer.add_graph(model, inputs)

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            sample_batch = inputs, labels
            meta = model.meta if n_gpu == 1 else model.module.meta
            if meta:
                outputs, labels = model(sample_batch)
            else:
                outputs = model(inputs)
            # preds_ref, acc_ref, labels, outputs = model(sample_batch)

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            probs = nn.functional.softmax(outputs, 1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Print batch result
            lr = [group["lr"] for group in optimizer.param_groups]
            if itr % print_per_iteration == 0:
                e_time = time.time()
                t = e_time - s_time
                acc = torch.sum(preds == labels.data).double() / len(preds)
                print("{:}: EPOCH{:03d} {:}Itr{:}/{:} ({:.2f}s/itr) Train: acc {:3.2f}, loss {:2.4f}, lr {:s}".format(
                    datetime.now(), epoch, int(print_per_iteration), itr // print_per_iteration,
                                                                     len(trainLoader) // print_per_iteration, t,
                    acc.item(),
                    loss.item(), str(lr)))

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Optimize
            loss.backward()
            optimizer.step()

        print("=" * 50)

        epoch_loss = running_loss / len(trainLoader.dataset)
        epoch_acc = running_corrects.double() / len(trainLoader.dataset)
        print("epoch {:d} | avg training loss {:.6f} | avg training acc {:.2f}".format(epoch, epoch_loss, epoch_acc))

        all_preds = np.concatenate(all_preds).reshape(-1)
        all_labels = np.concatenate(all_labels).reshape(-1)

        # Plot confusion matrix
        file_name = "Train_confusion_matrix"
        fig = plot_confusion_matrix(plot_folder, file_name, all_labels, all_preds, epoch, num_classes)
        writer.add_figure("train_confusion_matrix", fig, global_step=epoch)

        # Write to tensorboard
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)



        if testLoader:
            # Validation
            epoch_loss, epoch_acc = val(testLoader, model, criterion, plot_folder, epoch, num_classes)
            # Scheduler step
            scheduler.step(epoch_loss)

            # Write to tensorboard
            writer.add_scalar("Loss/test", epoch_loss, epoch)
            writer.add_scalar("Accuracy/test", epoch_acc, epoch)

            test_loss.append(epoch_loss)
            test_acc.append(epoch_acc)

            # if auc_score > best_auc_score or (auc_score == best_auc_score and val_accs > best_acc):
            if epoch_loss < best_val_loss or (epoch_loss == best_val_loss and epoch_acc > best_acc):
                save_path = os.path.join(save_dir, "epoch_" + str(epoch) + ".pt")
                torch.save(model.state_dict(), save_path)
                best_val_loss = epoch_loss
                best_acc = epoch_acc
                best_epoch = epoch + 1
                model_copy = copy.deepcopy(model)

        else:
            scheduler.step(epoch_loss)
        print("=" * 50)

    ### Retrain on whole train-val dataset using best_epoch number from pervious step
    # if trainValLoader is not None:
    #     # Retrain on train_val dataset
    #     model_copy = full_train(trainValLoader, augmentor, model_copy, optimizer, criterion, start_epoch, best_epoch,
    #            plot_folder, num_classes, print_per_iteration)
    #     torch.save(model_copy.state_dict(), os.path.join(save_dir, "full_epoch_" + str(best_epoch) + ".pt"))
    # else:
    #     model_copy = model

    if testLoader:
        from utils.plot_utils import plot_learning_cuvre
        plot_learning_cuvre(train_loss, test_loss, train_acc, test_acc, save_dir, epochs)


    return model_copy, best_epoch


def full_train(trainValLoader, augmentor, model_copy, optimizer, criterion, start_epoch, best_epoch,
               plot_folder, num_classes, print_per_iteration):
    for epoch in tqdm(range(start_epoch, best_epoch)):
        print("Epoch {}/{}".format(epoch + 1, best_epoch))
        print("-" * 10)

        # Training phase
        model_copy.train()
        all_preds = []
        all_probs = []
        all_labels = []

        running_loss = 0.0
        running_corrects = 0
        s_time = time.time()

        for itr, sample_batch in enumerate(trainValLoader):
            inputs, labels = sample_batch
            inputs = augmentor(inputs)
            if isinstance(inputs, list) or isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(np.array(inputs).astype(np.float32)).to(device)
                labels = torch.from_numpy(np.array(labels).astype(np.int)).to(device)
            else:
                inputs = inputs.float().to(device)
                labels = labels.to(device)

            # Write image to tensorboard
            if itr < 2:
                cube_size = inputs.shape[2]
                img_grid = torchvision.utils.make_grid(inputs[:, :, cube_size // 2])
                writer.add_image("full_train_images_itr{:d}".format(itr), img_grid, global_step=epoch)
                # writer.add_graph(model, inputs)

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model_copy(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            probs = nn.functional.softmax(outputs, 1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Print batch result
            lr = [group["lr"] for group in optimizer.param_groups]
            if itr % print_per_iteration == 0:
                e_time = time.time()
                t = e_time - s_time
                acc = torch.sum(preds == labels.data).double() / len(preds)
                print(
                    "TrainVal: {:}: EPOCH{:03d} {:}Itr{:}/{:} ({:.2f}s/itr) Train: acc {:3.2f}, loss {:2.4f}, lr {:s}".format(
                        datetime.now(), epoch, int(print_per_iteration), itr // print_per_iteration,
                                                                         len(trainValLoader) // print_per_iteration, t,
                        acc.item(), loss.item(), str(lr)))

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Optimize
            loss.backward()
            optimizer.step()

        print("=" * 50)

        epoch_loss = running_loss / len(trainValLoader.dataset)
        epoch_acc = running_corrects.double() / len(trainValLoader.dataset)
        print("TrainVal: epoch {:d} | avg training loss {:.6f} | avg training acc {:.2f}".format(epoch, epoch_loss,
                                                                                                 epoch_acc))

        all_preds = np.concatenate(all_preds).reshape(-1)
        all_labels = np.concatenate(all_labels).reshape(-1)

        # Plot confusion matrix
        file_name = "Full_train_confusion_matrix"
        fig = plot_confusion_matrix(plot_folder, file_name, all_labels, all_preds, epoch, num_classes)
        writer.add_figure(file_name, fig, global_step=epoch)

        # Write to tensorboard
        writer.add_scalar("Loss/full_train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/full_train", epoch_acc, epoch)


def val(testLoader, model, criterion, plot_folder, epoch, num_classes):
    # Run predict
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    running_loss = 0.0
    running_corrects = 0
    for sample_batch in testLoader:
        # Load inputs and labels
        inputs, labels = sample_batch
        if isinstance(inputs, list) or isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(np.array(inputs).astype(np.float32)).to(device)
            labels = torch.from_numpy(np.array(labels).astype(np.int)).to(device)
        else:
            inputs = inputs.float().to(device)
            labels = labels.to(device)

        # Forward pass
        sample_batch = inputs, labels
        # preds_ref, acc_ref, labels, outputs = model(sample_batch)
        meta = model.meta if n_gpu == 1 else model.module.meta
        if meta:
            outputs, labels = model(sample_batch)
        else:
            outputs = model(inputs)

        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        probs = nn.functional.softmax(outputs, 1)[:, 1]
        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(testLoader.dataset)
    epoch_acc = running_corrects.double() / len(testLoader.dataset)
    print("epoch {:d} | avg test loss {:.6f} | avg test acc {:.2f}".format(epoch, epoch_loss, epoch_acc))

    all_preds = np.concatenate(all_preds).reshape(-1)
    all_labels = np.concatenate(all_labels).reshape(-1)

    # Plot confusion matrix
    file_name = "Val_confusion_matrix"
    fig = plot_confusion_matrix(plot_folder, file_name, all_labels, all_preds, epoch, num_classes)

    # Add plot to tensorboard
    writer.add_figure(file_name, fig, global_step=epoch)

    return epoch_loss, epoch_acc

def test(testLoader, model, criterion, save_folder, plot_folder, epoch, num_classes):
    # Run predict
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    running_loss = 0.0
    running_corrects = 0
    for sample_batch in tqdm(testLoader):
        # Load inputs and labels
        inputs, labels = sample_batch
        if isinstance(inputs, list) or isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(np.array(inputs).astype(np.float32)).to(device)
            labels = torch.from_numpy(np.array(labels).astype(np.int)).to(device)
        else:
            inputs = inputs.float().to(device)
            labels = labels.to(device)
        # Forward pass
        sample_batch = inputs, labels
        # preds_ref, acc_ref, labels, outputs = model(sample_batch)
        meta = model.meta if n_gpu == 1 else model.module.meta
        if meta:
            outputs, labels = model(sample_batch)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        probs = nn.functional.softmax(outputs, 1)[:, 1]
        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(testLoader.dataset)
    epoch_acc = running_corrects.double() / len(testLoader.dataset)
    print("epoch {:d} | avg test loss {:.6f} | avg test acc {:.2f}".format(epoch, epoch_loss, epoch_acc))

    all_preds = np.concatenate(all_preds).reshape(-1)
    all_labels = np.concatenate(all_labels).reshape(-1)
    all_probs = np.concatenate(all_probs).reshape(-1)

    # Save predictions
    np.savez_compressed(os.path.join(save_folder, "preds.npz"), l=all_labels, p=all_probs)
    print("save prediction results to: {:s}".format(os.path.join(save_folder, "preds.npz")))

    # Plot confusion matrix
    file_name = "Test_confusion_matrix"
    plot_confusion_matrix(plot_folder, file_name, all_labels, all_preds, epoch, num_classes)


def test_loop(testLoader, model, criterion, save_dir, plot_dir, start_epoch, num_classes):
    """
    The normal test loop: test and cal the 0.95 mean_confidence_interval.
    """
    total_accuracy = 0.0
    test_epoch = 5
    total_h = np.zeros(test_epoch)
    total_accuracy_vector = []

    for epoch_idx in range(test_epoch):
        print("============ Testing on the test set ============")
        test(testLoader, model, criterion, save_dir, plot_dir, start_epoch, num_classes)

def merge_args(config, args):
    if args.data_dir is not None:
        config.DATA_DIR = args.data_dir
    if args.load_all is not None:
        config.LOAD_ALL = args.load_all
    # if args.pad_value is not None:
    #     config.PAD_VALUE = args.pad_value
    return config

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
    num_classes = args.n_classes
    optimStr = args.optimizer
    workers = args.workers
    test_size = args.test_size
    load_all = args.load_all

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
        config = merge_args(config, args)
        lungData = LungDataset(config)
        kfold = StratifiedKFold(n_splits=args.kfold, random_state=42) if kfold is not None else None
        # kfold = Kfold(n_splits=args.kfold, random_state=42) if kfold is not None else None
        datasets = lungData.get_datasets(kfold=kfold, splitId=splitId, loadAll=config.LOAD_ALL, test_size=test_size)
        # kfold = len(lungData.y) if kfold is None else args.kfold
        # kfold = KFold(n_splits=kfold, random_state=42)

        # trainLoader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate)
        trainLoader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=workers)
        if kfold is not None:
            trainValLoader = DataLoader(datasets["train_val"], batch_size=batch_size, shuffle=True, num_workers=workers)
        else:
            trainValLoader = None
        valLoader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=workers)
        testLoader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=workers)

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
        # from dataLoader.LUNA16Data import LUNA16, LunaConfig
        from dataLoader.LunaData import LunaDataset, LunaConfig

        config = LunaConfig()
        config = merge_args(config, args)
        lunaData = LunaDataset(config)
        kfold = StratifiedKFold(n_splits=args.kfold, random_state=42) if kfold is not None else None
        datasets = lunaData.get_datasets(kfold=kfold, splitId=splitId, loadAll=config.LOAD_ALL, test_size=test_size)
        # kfold = len(lungData.y) if kfold is None else args.kfold
        # kfold = KFold(n_splits=kfold, random_state=42)
        # trainLoader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate)
        trainLoader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=workers)
        if kfold is not None:
            trainValLoader = DataLoader(datasets["train_val"], batch_size=batch_size, shuffle=True, num_workers=workers)
        else:
            trainValLoader = None
        valLoader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=workers)
        testLoader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=workers)
        # root_dir = "../data/"
        # pos_label_file = "../data/pos_labels.csv"
        # cat_label_file = "../data/Lung Nodule Clinical Data_Min Kim (No name).xlsx"
        # load_model = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/model/classification_LUNA16/Resnet18_Adam_lr0.001"
        # cube_size = 48
        # trainData = LUNA16(train=True)
        # trainLoader = DataLoader(trainData, batch_size=2, shuffle=True)
        # valData = LUNA16(train=False)
        # valLoader = DataLoader(valData, batch_size=1, shuffle=False)

    print("Shape of train_x is: ", (len(datasets["train"]), 1,) + (config.CUBE_SIZE,) * 3)
    print("Shape of train_y is: ", (len(datasets["train"]),))
    print("Shape of val_x is: ", (len(datasets["val"]), 1,) + (config.CUBE_SIZE,) * 3)
    print("Shape of val_y is: ", (len(datasets["val"]),))

    # augmentor = augment(ifflip=config.FLIP, ifrotate=config.ROTATE, ifswap=config.SWAP)
    augmentor = augment()
    config.display()

    ## ----- Construct models ----- ##
    # Detect if we have a GPU available
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "res18":
        model = generate_model(18, n_input_channels=1, n_classes=num_classes, clinical=config.LOAD_CLINICAL)

        # model = ResNet_RFS(512, 2, device=device)
        # model = ResNet_MB(512, 2, device=device)

    print("Use model: {:s}".format(model_name))


    if load_model:
        if os.path.isfile(load_model):
            load_path = load_model
            start_epoch = int(os.path.basename(load_path)[:].strip(".pt")[6:])
        else:
            assert os.path.isdir(load_model), "Load_model {:} is either file or path".format(load_model)
            from natsort import natsorted
            model_list = [m for m in os.listdir(load_model) if m.endswith("pt")]
            model_list = natsorted(model_list)
            assert len(model_list) > 0, "No model listed in {:s}".format(load_model)
            load_path = os.path.join(load_model, model_list[-1])
            start_epoch = int(model_list[-1].strip(".pt")[6:])
        state_dict = torch.load(load_path)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                assert "module." in k
                name = k.replace("module.", "")
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            # ## Load emb_func only
            # for k, v in state_dict.items():
            #     # if "module." in k:
            #     #     name = k.replace("module.", "")
            #     # else:
            #     #     name = "module." + k  # add "module." for dataparallel
            #     name = k
            #     if "fc" in k:
            #         continue
            #         # name = name.replace("fc", "classifier")
            #     # if "emb_func" not in k and "fc" not in k:
            #     #     name = "emb_func." + name
            #     new_state_dict[name] = v
            # model.emb_func.load_state_dict(new_state_dict)

        print("Load successfully from " + load_path)
    else:
        start_epoch = 0
    if not args.resume:
        start_epoch = 0

    # Print the model we just instantiated
    model = model.to(device)
    print(model)

    freeze_bn(model) #!!!!!!!!

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

    if optimStr =="sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        assert optimStr == "adam"
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=10, min_lr=0.00001, patience=10)

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
    # model.load_state_dict(torch.load(os.path.join(load_model, "epoch_" + str(epoch) + ".pt")))
    # print("load model from: {:s}".format(os.path.join(load_model, "epoch_" + str(epoch) + ".pt")))


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
    plot_dir = os.path.join(save_dir, "test")
    os.makedirs(plot_dir, exist_ok=True)
    if train_flag:
        model, start_epoch = train(trainLoader, valLoader, trainValLoader, augmentor, model, optimizer, scheduler, criterion, save_dir, epochs, start_epoch)
        test(testLoader, model, criterion, save_dir, plot_dir, start_epoch, num_classes)
    else:
        test_loop(testLoader, model, criterion, save_dir, plot_dir, start_epoch, num_classes)
        # test(testLoader, model, criterion, save_dir, plot_dir, start_epoch, num_classes)




if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Spent {:.2f}s".format(time.time() - start_time))
