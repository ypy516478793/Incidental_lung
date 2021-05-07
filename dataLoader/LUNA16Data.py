from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

import numpy as np
import torch
import os

class LunaConfig(object):
    DATA_DIR = "../../"
    TRAIN_DATA_DIR = ['data/preprocessed/subset0/',
                      'data/preprocessed/subset1/',
                      'data/preprocessed/subset2/',
                      'data/preprocessed/subset3/',
                      'data/preprocessed/subset4/',
                      'data/preprocessed/subset5/',
                      'data/preprocessed/subset6/',
                      'data/preprocessed/subset7/']
    VAL_DATA_DIR = ['data/preprocessed/subset8/']
    TEST_DATA_DIR = ['data/preprocessed/subset9/']
    BLACK_LIST = []

    ANCHORS = [5., 10., 20.]
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
    PAD_VALUE = 170  # previous 170
    AUGTYPE = {"flip": True, "swap": False, "scale": True, "rotate": False}

    CONF_TH = 4
    NMS_TH = 0.3
    DETECT_TH = 0.5

    SIDE_LEN = 144
    MARGIN = 32

    ORIGIN_SCALE = False

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")



class LUNA16(Dataset):
    def __init__(self, rootFolder="/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/LUNA16/classification/", train=True):
        """
        House price dataset
        :param x: all_features, shape (#samples, #features)
        :param y: all_labels, shape (#samples,)
        :param transform: transform functions for features
        :param target_transform: transform functions for targets
        """

        posFolder = os.path.join(rootFolder, "1")
        negFolder = os.path.join(rootFolder, "0")
        self.allPosCases = os.listdir(posFolder)
        self.allNegCases = os.listdir(negFolder)
        num_pos = len(self.allPosCases)
        if len(self.allNegCases) > num_pos:
            self.allNegCases = np.random.choice(self.allNegCases, num_pos, replace=False)
        self.load_subset(train)
        assert len(self.posCases) == len(self.negCases)
        self.allPath = np.array([os.path.join(posFolder, p) for p in self.posCases] + \
                                [os.path.join(negFolder, n) for n in self.negCases])
        total_samples = len(self.allPath)
        self.allLabels = np.zeros([total_samples, 2], dtype=np.int)
        self.allLabels[:len(self.posCases), 0] = 1
        self.allLabels[len(self.posCases):, 1] = 1

        self.allLabels = np.argmax(self.allLabels, axis=-1)
        # shuffleIds = np.random.permutation(np.arange(total_samples))
        # self.allPath = self.allPath[shuffleIds]
        # self.allLabels = self.allLabels[shuffleIds]

    def load_subset(self, train):
        trainPosCases, valPosCases = train_test_split(self.allPosCases, test_size=0.4, random_state=42)
        trainNegCases, valNegCases = train_test_split(self.allNegCases, test_size=0.4, random_state=42)
        # trainInfo, valInfo = train_test_split(self.imageInfo)
        self.posCases = trainPosCases if train else valPosCases
        self.negCases = trainNegCases if train else valNegCases

    def __len__(self):
        return len(self.allPath)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.allPath[idx]
        feature = np.load(path)[np.newaxis, :]
        label = self.allLabels[idx]


        # if self.transform:
        #     feature = self.transform(feature)
        #
        # if self.target_transform:
        #     label = self.target_transform(label)

        sample = {"cubes": feature,
                  "label": label}

        return sample

if __name__ == '__main__':

    trainData = LUNA16(train=True)
    trainLoader = DataLoader(trainData, batch_size=2, shuffle=True)

    for sample_batch in trainLoader:
        x, y = sample_batch["cubes"], sample_batch["label"]
        print(x)
        print(y, "\n")