from tqdm import tqdm
from utils.model_utils import extract_cube, lumTrans
from utils.data_utils import balance_any_data
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np
import torch

import os


class IncidentalConfig(object):
    LOAD_ALL = False
    CROP_LUNG = True
    MASK_LUNG = True
    PET_CT = None
    # ROOT_DIR = "/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_kim/"
    # ROOT_DIR = "/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_Ben/"
    # ROOT_DIR = "/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_unlabeled/"
    # ROOT_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/maskCropDebug"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_unlabeled/masked_with_crop"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/labeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/unlabeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/raw_data/unlabeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_mamta/processed_data/unlabeled/"
    INFO_FILE = "CTinfo.npz"
    # POS_LABEL_FILE = "pos_labels_norm.csv"
    # POS_LABEL_FILE = "pos_labels_norm.csv"
    # POS_LABEL_FILE = "gt_labels_checklist.xlsx"
    # POS_LABEL_FILE = "Predicted_labels_checklist_Kim_TC.xlsx"
    # POS_LABEL_FILE = None
    # CAT_LABEL_FILE = None

    DATA_DIR = "./Methodist_incidental/data_Ben/modeNorm/"
    # DATA_DIR = "./Methodist_incidental/data_Ben/resampled/"
    POS_LABEL_FILE = "./Methodist_incidental/data_Ben/resampled/pos_labels_norm.csv"
    CAT_LABEL_FILE = "./Methodist_incidental/Lung Nodule Clinical Data_Min Kim - Added Variables 10-2-2020.xlsx"
    # CAT_LABEL_FILE = None

    BLACK_LIST = ["001030196-20121205", "005520101-20130316", "009453325-20130820", "034276428-20131212",
                  "036568905-20150714", "038654273-20160324", "011389806-20160907", "015995871-20160929",
                  "052393550-20161208", "033204314-20170207", "017478009-20170616", "027456904-20180209",
                  "041293960-20170227", "000033167-20131213", "022528020-20180525", "025432105-20180730",
                  "000361956-20180625"]
    LOAD_CLINICAL = False

    CUBE_SIZE = 64

    ANCHORS = [10.0, 30.0, 60.0]
    MAX_NODULE_SIZE = 60
    # ANCHORS = [5., 10., 20.]  # [ 10.0, 30.0, 60.]
    CHANNEL = 1
    CROP_SIZE = [96, 96, 96]
    STRIDE = 4
    MAX_STRIDE = 16
    NUM_NEG = 800
    TH_NEG = 0.02
    TH_POS_TRAIN = 0.5
    TH_POS_VAL = 1
    NUM_HARD = 2
    BOUND_SIZE = 12
    RESO = 1
    SIZE_LIM = 2.5  # 3 #6. #mm
    SIZE_LIM2 = 10  # 30
    SIZE_LIM3 = 20  # 40
    AUG_SCALE = True
    R_RAND_CROP = 0.3
    PAD_VALUE = 0   # previous 170
    # AUGTYPE = {"flip": False, "swap": False, "scale": False, "rotate": False, "contrast": False, "bright": False, "sharp": False, "splice": False}
    # AUGTYPE = {"flip": True, "swap": True, "scale": True, "rotate": True}

    KFOLD = None
    KFOLD_SEED = None

    FLIP = False
    SWAP = False
    SCALE = False
    ROTATE = False
    CONSTRAST = False
    BRIGHT = False
    SHARP = False
    SPLICE = False



    CONF_TH = 4
    NMS_TH = 0.3
    DETECT_TH = 0.5

    SIDE_LEN = 144
    MARGIN = 32

    ORIGIN_SCALE = False
    SPLIT_SEED = 42
    LIMIT_TRAIN = None
    SPLIT_ID = None

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

class Base(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = self.images[item]
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)



class LungDataset(object):
    # def __init__(self, rootFolder, pos_label_file=None, cat_label_file=None, cube_size=64,
    #              train=None, screen=True, clinical=False):
    # def __init__(self, root_dir, POS_LABEL_FILE, CAT_LABEL_FILE, config, subset="train"):
    def __init__(self, config):
        self.config = config
        self.data_dir = config.DATA_DIR
        self._imageIds = []
        # Load position label file
        pos_label_file = config.POS_LABEL_FILE
        if pos_label_file is not None:
            self.pos_df = pd.read_csv(pos_label_file, dtype={"date": str})
        # Load category label file
        cat_label_file = config.CAT_LABEL_FILE
        if cat_label_file:
            self.cat_df = pd.read_excel(cat_label_file, dtype={"MRN": str}, sheet_name='Sheet1')
            self.additional_df = pd.read_excel(cat_label_file, dtype={"MRN": str}, sheet_name='Sheet2')
            cat_key = [i for i in self.cat_df.columns if i.startswith("Category Of")][0]
            self.cats = self.cat_df[cat_key]
        # Process clnical information
        if config.LOAD_CLINICAL:
            self.clinical_preprocessing()
        # Process imageInfo (Load subset)
        self.imageInfo = np.load(os.path.join(config.DATA_DIR, "CTinfo.npz"), allow_pickle=True)["info"]
        self.imageInfo = np.array(self.imageInfo)
        self.__remove_duplicate__()
        self.__check_labels__()
        self.__screen__()
        self.load_data()

    def get_datasets(self, kfold=None, splitId=None, loadAll=False, test_size=0.1):
        if loadAll:
            datasets_dict = {"train": Base(self.X, self.y),
                             "val": Base(self.X, self.y),
                             "test": Base(self.X, self.y)}
            return datasets_dict

        datasets = self.load_subset(random_state=self.config.SPLIT_SEED, kfold=kfold, splitId=splitId, test_size=test_size)
        datasets_dict = {}
        for subset in ["train", "val", "test", "train_val"]:
            if subset == "train_val":
                images = np.concatenate([datasets["train"]["X"], datasets["train"]["X"]])
                labels = np.concatenate([datasets["train"]["y"], datasets["train"]["y"]])
                datasets_dict[subset] = Base(images, labels)
            else:
                dataset = datasets[subset]
                datasets_dict[subset] = Base(dataset["X"], dataset["y"])
        return datasets_dict


    # def load_subset(self, random_state=42, kfold=None, splitId=None):
    #     datasets = {}
    #     if kfold is None:
    #         X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=random_state)
    #         X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
    #     else:
    #         assert splitId is not None
    #         all_indices = np.arange(len(self.X))
    #         kf_indices = [(train_index, test_index) for train_index, test_index in kfold.split(all_indices)]
    #         train_index, test_index = kf_indices[splitId]
    #         X_train, X_test = self.X[train_index], self.X[test_index]
    #         y_train, y_test = self.y[train_index], self.y[test_index]
    #         X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1,
    #                                                           random_state=random_state)
    #         print("X_train size: ", X_train.shape)
    #         print("X_val size: ", X_val.shape)
    #         print("X_test size: ", X_test.shape)
    #
    #     X_train, y_train = balance_any_data(X_train, y_train)
    #     X_val, y_val = balance_any_data(X_val, y_val)
    #     datasets["train"] = {"X": X_train, "y": y_train}
    #     datasets["val"] = {"X": X_val, "y": y_val}
    #     datasets["test"] = {"X": X_test, "y": y_test}
    #     return datasets


    def load_subset(self, random_state=42, kfold=None, splitId=None, test_size=0.1):
        datasets = {}
        if kfold is None:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=random_state)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
        else:
            assert splitId is not None
            all_indices = np.arange(len(self.X))
            kf_indices = [(train_index, test_index) for train_index, test_index in kfold.split(all_indices, self.y)]
            train_index, test_index = kf_indices[splitId]
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size,
                                                              random_state=random_state)
            print("X_train size: ", X_train.shape)
            print("X_val size: ", X_val.shape)
            print("X_test size: ", X_test.shape)

        X_train, y_train = balance_any_data(X_train, y_train)
        X_val, y_val = balance_any_data(X_val, y_val)
        datasets["train"] = {"X": X_train, "y": y_train}
        datasets["val"] = {"X": X_val, "y": y_val}
        datasets["test"] = {"X": X_test, "y": y_test}
        return datasets

    # def split_data(self, kfold=None, splitId=None):
    #     if args.balance_option == "before":
    #         self.X, self.y = self.balance_any_data(self.X, self.y)
    #     self.X = self.X[..., np.newaxis] / 255.0
    #     self.X = np.repeat(self.X, 3, axis=-1)
    #     if kfold is None:
    #         X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
    #     else:
    #         assert splitId is not None
    #         all_indices = np.arange(len(self.X))
    #         kf_indices = [(train_index, val_index) for train_index, val_index in kfold.split(all_indices)]
    #         train_index, val_index = kf_indices[splitId]
    #         X_train, X_test = self.X[train_index], self.X[val_index]
    #         y_train, y_test = self.y[train_index], self.y[val_index]
    #
    #     if args.balance_option == "after":
    #         X_train, y_train = self.balance_any_data(X_train, y_train)
    #
    #     # X_train = X_train[:24]
    #     # y_train = y_train[:24]
    #
    #     self.train_data = (X_train, y_train)
    #     self.test_data = (X_test, y_test)
    #
    #     print("Shape of train_x is: ", X_train.shape)
    #     print("Shape of train_y is: ", y_train.shape)
    #     print("Shape of test_x is: ", X_test.shape)
    #     print("Shape of test_y is: ", y_test.shape)

    def load_data(self, reload=False):
        # data_path = os.path.join(self.data_dir, "3D_incidental_lung_multiNeg.npz")
        data_path = os.path.join(self.data_dir, "Methodist_3Dcubes_p64.npz")
        if os.path.exists(data_path) and not reload:
            self.data = np.load(data_path, allow_pickle=True)
            self.X, self.y = self.data["x"], self.data["y"]
        else:
            self.load_raw(data_path)

    def load_raw(self, data_path):
        print("Preprocessing -- Crop cubes")
        X, y = [], []
        for imageId in tqdm(range(len(self.imageInfo))):
            # image = self.load_image(imageId)
            # pos = self.load_pos(imageId)
            # assert len(pos) > 0, "Error: no data!"
            cubes = self.get_cube(imageId, self.config.CUBE_SIZE)
            label = self.load_cat(imageId)
            labels = np.repeat(label, len(cubes))

            # slices = self.get_slices(i, self.image_size)
            # label = self.load_cat(i)
            # labels = np.eye(self.num_classes, dtype=np.int)[np.repeat(label, len(slices))]
            X.append(cubes)
            y.append(labels)
        self.X = np.expand_dims(np.concatenate(X), axis=1)
        self.y = np.concatenate(y)

        np.savez_compressed(data_path, x=self.X, y=self.y)
        print("Save slice 3D incidental lung nodule data to {:s}".format(data_path))

    def __remove_duplicate__(self):
        for i, info in enumerate(self.imageInfo):
            if info["date"] == "":
                info["date"] = info["imagePath"].strip(".npz").split("-")[-1]

        identifier_set = ["{:}-{:}".format(info["patientID"], info["date"]) for info in self.imageInfo]
        remove_ids = []
        from collections import Counter
        cnt = Counter(identifier_set)
        for k, v in cnt.items():
            if k in self.config.BLACK_LIST:
                indices = [i for i, x in enumerate(identifier_set) if x == k]
                remove_ids = remove_ids + indices
            elif v > 1:
                indices = [i for i, x in enumerate(identifier_set) if x == k]
                remove_ids = remove_ids + indices[:-1]
        self.imageInfo = np.delete(self.imageInfo, remove_ids)

    def __check_labels__(self):
        for info in tqdm(self.imageInfo):
            pstr = info["pstr"]
            dstr = info["date"]
            existId = (self.pos_df["patient"] == pstr) & (self.pos_df["date"] == dstr)
            assert existId.sum() > 0, "no matches, pstr {:}, dstr {:}".format(pstr, dstr)

    # def load_subset(self, subset, random_state=None, limit_train_size=None, kfold=None, splitId=None):
    #     if subset == "inference":
    #         infos = self.imageInfo
    #     else:
    #         ## train/val/test split
    #         if random_state is None:
    #             random_state = 42
    #         if kfold is None:
    #             trainInfo, valInfo = train_test_split(self.imageInfo, test_size=0.6, random_state=random_state)
    #             valInfo, testInfo = train_test_split(valInfo, test_size=0.5, random_state=random_state)
    #         else:
    #             assert splitId is not None
    #             trainValInfo, testInfo = train_test_split(self.imageInfo, test_size=0.2, random_state=random_state)
    #             kf_indices = [(train_index, val_index) for train_index, val_index in kfold.split(trainValInfo)]
    #             train_index, val_index = kf_indices[splitId]
    #             trainInfo, valInfo = trainValInfo[train_index], trainValInfo[val_index]
    #
    #
    #         assert subset == "train" or subset == "val" or subset == "test" or subset =="train_val", "Unknown subset!"
    #         if subset == "train":
    #             infos = trainInfo
    #             if limit_train_size is not None:
    #                 infos = infos[:int(limit_train_size * len(infos))]
    #         elif subset == "val":
    #             infos = valInfo
    #         elif subset == "train_val":
    #             infos = np.concatenate([trainInfo, valInfo])
    #         else:
    #             infos = testInfo
    #     self.imageInfo = infos
    #
    # def prepare(self):
    #     self.num_images = len(self.imageInfo)
    #     self._imageIds = np.arange(self.num_images)
    #     self.patient2Image = {"{:s}-{:s}".format(info['patientID'], info['date']): id
    #                                   for info, id in zip(self.imageInfo, self.imageIds)}
    #     self.patient2Image.update({"{:s}-{:s}".format(info['pstr'], info['date']): id
    #                               for info, id in zip(self.imageInfo, self.imageIds)})
    # @property
    # def imageIds(self):
    #     return self._imageIds

    def __screen__(self):
        num_images = len(self.imageInfo)
        mask = np.ones(num_images, dtype=bool)
        for imageId in range(num_images):
            pos = self.load_pos(imageId)
            cat = self.load_cat(imageId)
            # if len(pos) > 1:
            if len(pos) > 1 and cat == 0:
                mask[imageId] = False
        self.imageInfo = self.imageInfo[mask]

    def clinical_preprocessing(self):
        dropCols = ["Sex"]
        for col in self.additional_df.columns[2:]:
            if self.additional_df[col].isnull().values.all():
                dropCols.append(col)
        self.additional_df = self.additional_df.drop(columns=dropCols)
        self.additional_df = self.additional_df.replace({"Y": 1, "N": 0})
        self.additional_df = self.additional_df.fillna(-1)
        for col in ["Race", "Ethnicity", "Insurance"]:
            self.additional_df[col] = self.additional_df[col].astype("category").cat.codes

        newOrder = []
        for i in range(len(self.cat_df)):
            MRN = self.cat_df.iloc[i]["MRN"]
            DOS = self.cat_df.iloc[i]["Date Of Surgery {1340}"]
            existId = (self.additional_df["MRN"] == MRN) & (self.additional_df["Date Of Surgery {1340}"] == DOS)
            idx = self.additional_df[existId].index
            assert len(idx) == 1
            newOrder.append(idx[0])
        self.reOrderAdditional_df = self.additional_df.reindex(newOrder).reset_index(drop=True)
        self.cat_df = pd.concat([self.cat_df, self.reOrderAdditional_df.iloc[:, 2:]], axis=1)

        dropCols = ["Patient index",
                    "Annotation meta info",
                    "Date Of Surgery {1340}",
                    "Date of Birth",
                    "Category Of Disease - Primary {1300} (1=lung cancer, 2=metastatic, 3 = benign nodule, 4= bronchiectasis/pulm sequestration/infection)",
                    "Date Of Surgery {1340}2",
                    "Pathologic Staging - Lung Cancer - T {1540}",
                    "Pathologic Staging - Lung Cancer - N {1550}",
                    "Pathologic Staging - Lung Cancer - M {1560}",
                    "Lung Cancer - Number of Nodes {1570}"]
        self.cat_df = self.cat_df.drop(columns=dropCols)
        self.cat_df = self.cat_df.replace({"Yes": 1, "No": 0})
        self.cat_df = self.cat_df.fillna(-1)
        self.cat_df["Race Documented {191}"] = self.cat_df["Race Documented {191}"].replace("Patient declined to disclose", -1)
        self.cat_df["Cerebrovascular History {620}"] = self.cat_df["Cerebrovascular History {620}"].astype("category").cat.codes
        self.cat_df["ASA Classification {1470}"] = self.cat_df["ASA Classification {1470}"].replace({"II": 2, "III":3, "IV": 4})
        self.cat_df["Cigarette Smoking {730}"] = self.cat_df["Cigarette Smoking {730}"].astype("category").cat.codes
        from sklearn import preprocessing
        StandardScaler = preprocessing.StandardScaler()
        dataCols = self.cat_df.columns[1:]
        cat_scaled = StandardScaler.fit_transform(self.cat_df[dataCols])
        data_df = pd.DataFrame(cat_scaled, columns=dataCols)
        self.cat_df = pd.concat([self.cat_df.iloc[:, 0], data_df], axis=1)

    def load_image(self, imageId):
        imgInfo = self.imageInfo[imageId]
        imgPath, thickness, spacing = imgInfo["imagePath"], imgInfo["sliceThickness"], imgInfo["pixelSpacing"]
        images = np.load(imgPath)["image"]
        images = lumTrans(images)

        # masked = np.array([make_lungmask(i, display=True) for i in images])

        # if mask:
        #     masked_images = []
        #     for img in images:
        #         masked_images.append(make_lungmask(img))
        #     masked_images = np.stack(masked_images)
        # plt.imshow(images[10])
        # print("Images{:d} shape: ".format(imageId), images.shape)

        return images

    def load_pos(self, imageId):
        imgInfo = self.imageInfo[imageId]
        pstr = imgInfo["pstr"]
        dstr = imgInfo["date"]
        patient_colname = "patient" if "patient" in self.pos_df.columns else 'Patient\n Index'
        assert patient_colname in self.pos_df
        existId = (self.pos_df[patient_colname] == pstr) & (self.pos_df["date"] == dstr)
        pos = self.pos_df[existId][["x", "y", "z", "d"]].values

        return pos

    def load_cat(self, imageId):
        imgInfo = self.imageInfo[imageId]
        patientID = imgInfo["patientID"]
        existId = (self.cat_df["MRN"].str.zfill(9) == patientID)
        cat = self.cats[existId].iloc[0]
        cat = int(cat <= 2)  # 1=lung cancer, 2=metastatic, 3 = benign nodule, 4= bronchiectasis/pulm sequestration/infection
        return cat

    def get_clinical(self, imageId):
        imgInfo = self.imageInfo[imageId]
        pid = int(imgInfo["pstr"][-3:])
        assert self.cat_df.iloc[pid-1]["MRN"].zfill(9) == imgInfo["patientID"]
        clinical_info = self.cat_df.iloc[pid-1]
        return clinical_info.values[1:]

    def get_cube(self, imageId, size):
        imgInfo = self.imageInfo[imageId]
        pos = self.load_pos(imageId)
        cubes = []
        for i,p in enumerate(pos):
            cubePath = imgInfo["imagePath"].replace(".npz", "_cube{:d}_{:d}.npz".format(size, i))
            try:
                cube = np.load(cubePath, allow_pickle=True)["image"]
            except FileNotFoundError:
                images = self.load_image(imageId)
                cube = extract_cube(images, p, size=size)
                # np.savez_compressed(cubePath, image=cube, info=imgInfo, pos=p)
                # print("Save scan cube to {:s}".format(cubePath))
            cubes.append(cube)
        cubes = np.array(cubes)

        return cubes


if __name__ == '__main__':
    config = IncidentalConfig()
    lungData = LungDataset(config)
    datasets = lungData.get_datasets()
    from torch.utils.data import DataLoader
    trainLoader = DataLoader(datasets["train"], batch_size=4, shuffle=True)
    for sample_batch in trainLoader:
        # inputs, labels = sample_batch["cubes"], sample_batch["label"]
        inputs, labels = sample_batch

    # writer = SummaryWriter(os.path.join("Visualize", "MethodistFull"))
    # config = IncidentalConfig()

    # # rootFolder = "/Users/yuan_pengyu/Downloads/IncidentalLungCTs_sample/"
    # # rootFolder = "data/"
    # rootFolder = "data_king/labeled/"
    # # rootFolder = "data_king/unlabeled/"
    # pos_label_file = "data/pos_labels.csv"
    # cat_label_file = "data/Lung Nodule Clinical Data_Min Kim - Added Variables 10-2-2020.xlsx"
    # cube_size = 64
    # lungData = LungDataset(rootFolder, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
    #                        cube_size=cube_size, train=None, screen=False, clinical=False)
    # # image, new_image = lungData.load_image(0)
    # # img = new_image[100]
    # # make_lungmask(img, display=True)
    #
    # # from prepare_lung import show_nodules
    # # crop_size = 64
    # # show_nodules(lungData, crop_size)
    # #
    # # lungData_test = LungDataset(rootFolder, labeled_only=True, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
    # #                        cube_size=cube_size, reload=False, train=False)
    # # crop_size = 64
    # # show_nodules(lungData_test, crop_size, train=False)
    #
    #
    #
    # # saveFolder = "./data/"
    # # for id in tqdm(lungData.imageIds):
    # #     image, new_image = lungData.load_image(id)
    # #     masked_lung = []
    # #     for img in new_image:
    # #         masked_lung.append(make_lungmask(img))
    # #     masked_lung = np.array(masked_lung)
    # #     fileName = "CT_scan_{:d}".format(id)
    # #     np.save(os.path.join(saveFolder, fileName + "_clean.npy"), masked_lung)
    # #     np.save(os.path.join(saveFolder, fileName + "_label.npy"), np.array([]))
    # #     print("Save data_{:d} to {:s}".format(id, os.path.join(saveFolder, fileName + "_clean.npy")))
    #
    # from torch.utils.data import DataLoader
    # from utils import collate
    # dataLoader = DataLoader(lungData, batch_size=2, drop_last=False, collate_fn=collate)
    # for sample in dataLoader:
    #     image, cubes, label = sample["image"], sample["cubes"], sample["label"]
    #     print("")
    #
    # print("")

    # # ## --------------- mannually check labels --------------- ##
    # rootFolder = "data_king/labeled/"
    # pos_label_file = "data/pos_labels.csv"
    # cat_label_file = "data/Lung Nodule Clinical Data_Min Kim - Added Variables 10-2-2020.xlsx"
    # cube_size = 64
    # lungData = LungDataset(rootFolder, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
    #                        cube_size=cube_size, train=None, screen=False, clinical=False)
    # lungData.pos_df2 = pd.read_excel("data_king/gt_labels_checklist.xlsx", sheet_name="confident_labels_checklist",
    #                                  skiprows=1, dtype={"date": str})
    # from utils import plot_bbox
    # def plot_individual_nodule(i, nodule_idx=0, new_d=None, new_loc=None):
    #     imgs = lungData.load_image(i)
    #     imgInfo = lungData.imageInfo[i]
    #     for k, v in imgInfo.items():
    #         print(k, ": ", v)
    #     thickness, spacing = imgInfo["sliceThickness"], imgInfo["pixelSpacing"]
    #     pstr = imgInfo["pstr"]
    #     dstr = imgInfo["date"]
    #     existId = (lungData.pos_df["patient"] == pstr) & (lungData.pos_df["date"] == dstr)
    #     pos = lungData.pos_df[existId][["x", "y", "z", "d"]].values
    #     print("len of pos: ", len(pos))
    #     if new_d is not None: pos[nodule_idx, 3] = new_d
    #     if new_loc is not None: pos[nodule_idx, :3] = new_loc
    #     pos[:, 2] = pos[:, 2] - 1
    #     print("original pos: ")
    #     print(np.array2string(pos, separator=', '))
    #     pos = np.array([resample_pos(p, thickness, spacing, imgshape=imgs.shape) for p in pos])
    #     print("isotropic pos: ")
    #     print(np.array2string(pos, separator=', '))
    #
    #     existId = (lungData.pos_df2["Patient\n Index"] == pstr) & (lungData.pos_df2["date"] == dstr)
    #     pos2 = lungData.pos_df2[existId][["x", "y", "z", "d"]].values
    #     print("original pos from new labels: ")
    #     print(np.array2string(pos2, separator=', '))
    #     size = lungData.pos_df2[existId]["size(mm)"].values
    #     print("real size: ", size)
    #     plot_bbox(imgs, pos[nodule_idx], None)
    #
    #     return imgs, pos[nodule_idx]
    #
    # a, b = plot_individual_nodule(63, nodule_idx=1, new_d=None, new_loc=None)


    ## --------------- save central slices with original size --------------- ##
    # from PIL import Image
    # from utils import plot_bbox
    # rootFolder = "data_king/labeled/"
    # pos_label_file = "data/pos_labels.csv"
    # cat_label_file = "data/Lung Nodule Clinical Data_Min Kim - Added Variables 10-2-2020.xlsx"
    # cube_size = 64
    # lungData = LungDataset(rootFolder, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
    #                        cube_size=cube_size, train=None, screen=False, clinical=False)
    # saveDir = "isotropic_central_slices"
    # os.makedirs(saveDir, exist_ok=True)
    # for i in tqdm(range(62, len(lungData.imageInfo))):
    #     info = lungData.imageInfo[i]
    #     save_name = "{:s}_{:s}_{:s}".format(info["pstr"], info["patientID"], info["date"])
    #     save_name += "_PET" if info["PET"] == "Y" else ""
    #     img = lungData.load_image(i)
    #     pos = lungData.load_pos(i)
    #     for idx, p in enumerate(pos):
    #         z = int(p[2])
    #         c_img = img[z]
    #         save_dir = os.path.join(saveDir, save_name + "_no{:d}_z{:d}.png".format(idx, z))
    #         PILimg = Image.fromarray(c_img)
    #         PILimg.save(save_dir)
    #         bbox_save_dir = save_dir.replace(".png", "_bbox.png")
    #         plot_bbox(img, p, bbox_save_dir, show=False)

## --------------- step by step --------------- ##

# saveFolder = "/Users/yuan_pengyu/Shared/Incidental_lung_nodule/Report"
# sample_stack(image, rows=4, cols=6)
# print("Shape before resampling: ", image.shape)
# plt.savefig(os.path.join(saveFolder, "beforeResampling.png"), bbox_inches="tight", dpi=200)
#
# sample_stack(new_image, rows=4, cols=6, show_every=10)
# print("Shape after resampling: ", new_image.shape)
# plt.savefig(os.path.join(saveFolder, "afterResampling.png"), bbox_inches="tight", dpi=200)


# saveFolder = "/Users/yuan_pengyu/Shared/Incidental_lung_nodule/Report"
# plot_hist(img)
# plt.savefig(os.path.join(saveFolder, "beforeNorm.png"), bbox_inches="tight", dpi=200)
# plot_hist(img)
# plt.savefig(os.path.join(saveFolder, "afterNorm.png"), bbox_inches="tight", dpi=200)
#
#
# plt.figure(); plt.imshow(img, cmap="gray")
# plt.savefig(os.path.join(saveFolder, "original.png"), bbox_inches="tight", dpi=200)
# plt.figure(); plt.imshow(middle, cmap="gray")
# plt.savefig(os.path.join(saveFolder, "middle.png"), bbox_inches="tight", dpi=200)
#
#
# plot_hist(middle)
# plt.plot([threshold, threshold], [0, 7500], "r--")
# plt.ylim([0, 7500])
# plt.savefig(os.path.join(saveFolder, "middle.png"), bbox_inches="tight", dpi=200)
#
# plt.figure(); plt.imshow(thresh_img, cmap="gray")
# plt.savefig(os.path.join(saveFolder, "thresh_img.png"), bbox_inches="tight", dpi=200)
# plt.figure(); plt.imshow(eroded, cmap="gray")
# plt.savefig(os.path.join(saveFolder, "eroded.png"), bbox_inches="tight", dpi=200)
# plt.figure(); plt.imshow(dilation, cmap="gray")
# plt.savefig(os.path.join(saveFolder, "dilation.png"), bbox_inches="tight", dpi=200)
#
# plt.figure(); plt.imshow(labels)
# plt.savefig(os.path.join(saveFolder, "labels.png"), bbox_inches="tight", dpi=200)
#
# import matplotlib.patches as patches
# plt.figure(); plt.imshow(labels)
# ax = plt.gca()
# rect = patches.Rectangle((B[1],B[0]),B[3]-B[1],B[2]-B[0],linewidth=1,edgecolor='r',facecolor='none')
# # Add the patch to the Axes
# ax.add_patch(rect)
# plt.savefig(os.path.join(saveFolder, "lung.png"), bbox_inches="tight", dpi=200)
#
# plt.figure(); plt.imshow(mask, cmap="gray")
# plt.savefig(os.path.join(saveFolder, "mask.png"), bbox_inches="tight", dpi=200)
#
# plt.figure(); plt.imshow(mask * img, cmap="gray")
# plt.savefig(os.path.join(saveFolder, "final.png"), bbox_inches="tight", dpi=200)


## --------------- preprocessing for one scan --------------- ##

# saveFolder = "/Users/yuan_pengyu/Shared/Incidental_lung_nodule/Report"
# masked_lung = []
# for img in new_image:
#     masked_lung.append(make_lungmask(img))
# sample_stack(masked_lung, rows=4, cols=6, show_every=10)
# plt.savefig(os.path.join(saveFolder, "afterPreprocessing.png"), bbox_inches="tight", dpi=200)





# ## --------------- show all images --------------- ##
#
# def plot_no_border(fileName, image, dpi=200, scale=2):
#     h, w = image.shape[:2]
#     fig, ax = plt.subplots(1, figsize=(w/dpi*scale,h/dpi*scale))
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     fig.add_axes(ax)
#     ax.imshow(image, cmap="gray")
#     fig.savefig(fileName, dpi=dpi)
#     plt.close(fig)
#
# def makedir(folder):
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#
# import matplotlib.patches as patches
#
# def plot_nodule(scan, label, sliceThickness, pixelSpacing, labelFolder):
#     '''
#
#     :param scan:
#     :param label: shape (z, x, y, d); (sliceId, pixel, pixel, mm)
#     :param sliceThickness: CT scan slice thickness (mm/slice)
#     :param pixelSpacing: pixel spacing in each CT scan (mm/pixel)
#     :return:
#     '''
#     z, x, y, d = label
#     sliceD = d/sliceThickness
#     pixelDx = d/pixelSpacing[0]
#     pixelDy = d/pixelSpacing[1]
#
#     minz, maxz = z - sliceD, z + sliceD + 0.01
#     for i in range(len(scan)):
#         if minz <= i and i <= maxz:
#             fig, ax = plt.subplots(1)
#             ax.imshow(scan[i], cmap="gray")
#             rect = patches.Rectangle((x - pixelDx / 2, y - pixelDy / 2), pixelDx, pixelDy, linewidth=1, edgecolor='r', facecolor='none')
#             ax.add_patch(rect)
#             plt.savefig(os.path.join(labelFolder, "{:d}_labeled.png".format(i)), dpi=200, bbox_inches="tight")
#     plt.show()
#
# # step 1: save images
# ID = 0
# scale = 2
# dmin = 3
# dmax = 4
# image, new_image = lungData.load_image(ID)
# saveFolder = os.path.dirname(lungData.imageInfo[ID]["imagePath"])
# imageFolder = os.path.join(saveFolder, "Images")
# makedir(imageFolder)
#
# for i in range(len(image)):
#     plot_no_border(os.path.join(imageFolder, "{:d}.png".format(i)), image[i], scale=2)
#
# sliceThickness = float(lungData.imageInfo[ID]["sliceThickness"])
# pixelSpacing = [float(s) for s in lungData.imageInfo[ID]["pixelSpacing"]]
# print("min bbox size: ", dmin/pixelSpacing[0]*scale, dmin/pixelSpacing[1]*scale)
# print("max bbox size: ", dmax/pixelSpacing[0]*scale, dmax/pixelSpacing[1]*scale)
#
#
# # step 2: plot nodule bbox
# sloc = 19
# xyloc = (640, 500)
# d = 4
# label = [sloc, xyloc[0]/scale, xyloc[1]/scale, d]
#
# labelFolder = os.path.join(saveFolder, "Label")
# makedir(labelFolder)
# with open(os.path.join(labelFolder, "label.csv"), "w") as f:
#     line = ','.join(map(str, label)) + "  # shape (z, x, y, d); (sliceId, pixel, pixel, mm)"
#     f.write(line)
#
# plot_nodule(image, label, sliceThickness, pixelSpacing, labelFolder)
#

