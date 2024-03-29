from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
from privacy.Model_Inversion_Attack.utils_test import check_fileType, read_slices, load_dicom, extract_cube, resample_image, resample_pos, make_lungmask, lumTrans
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pydicom as dicom
import pandas as pd
import numpy as np

from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

import os

import h5py


class LungDataset(Dataset):
    def __init__(self, rootFolder, cat_label_file=None, cube_size=64, train=None, clinical=False):
        self.imageInfo = []
        self._imageIds = []
        self.cube_size = cube_size
        if cat_label_file:
            self.cat_df = pd.read_excel(cat_label_file)
            self.cat_df = self.cat_df.drop([10, 11]).reset_index(drop=True)
            cat_key = [i for i in self.cat_df.columns if i.startswith("Category Of")][0]
            self.cats = self.cat_df[cat_key]
        self.clinical_preprocessing()
        self.load_clinical = clinical
        self.matches = ["LUNG", "lung"]
        self.load_lung(rootFolder)
        if train is not None:
            self.load_subset(train)
        self.prepare()

    def __len__(self):
        return len(self.imageIds)

    def __getitem__(self, imageId):
        if torch.is_tensor(imageId):
            imageId = imageId.tolist()

        if self.load_clinical:
            feature = self.get_clinical(imageId).astype(np.float32)
        else:
            feature = self.get_cube(imageId, self.cube_size)

        # feature = feature[np.newaxis, ...]
        label = self.load_cat(imageId)

        sample = {"features": feature,
                  "label": label}

        return sample

    def add_scan(self, pstr, imgPath, **kwargs):
        '''
        Add current scan meta information into global list
        :param: meta information for current scan
        :return: scan_info (in dictionary)
        '''
        scanInfo = {
            "pstr": pstr,
            "imagePath": imgPath,
        }
        scanInfo.update(kwargs)
        self.imageInfo.append(scanInfo)
        return scanInfo

    def load_from_dicom(self, rootFolder):
        '''
        load image from dicom files
        :param rootFolder: root folder of the data
        :return: None
        '''
        pos_patients = natsorted([i for i in os.listdir(os.path.join(rootFolder, "posCases"))])
        neg_patients = natsorted([i for i in os.listdir(os.path.join(rootFolder, "negCases"))])

        for i in range(len(pos_patients)):
            pstr = pos_patients[i].split(".")[0]
            imagePath = os.path.join(rootFolder, "posCases", pos_patients[i])
            scanInfo = self.add_scan(pstr, imagePath)

        for i in range(len(neg_patients)):
            pstr = neg_patients[i].split(".")[0]
            imagePath = os.path.join(rootFolder, "negCases", neg_patients[i])
            scanInfo = self.add_scan(pstr, imagePath)

        # print "-" * 30 + " CTinfo " + "-" * 30
        # [print(i) for i in self.imageInfo]

    def load_lung(self, rootFolder):
        self.load_from_dicom(rootFolder)
        self.imageInfo = np.array(self.imageInfo)

    def load_subset(self, train):
        trainInfo, valInfo = train_test_split(self.imageInfo, random_state=42)
        self.imageInfo = trainInfo if train else valInfo

    def prepare(self):
        self.num_images = len(self.imageInfo)
        self._imageIds = np.arange(self.num_images)
        self.patient2Image = {"{:s}".format(info['pstr']): id
                                      for info, id in zip(self.imageInfo, self.imageIds)}
    @property
    def imageIds(self):
        return self._imageIds

    def clinical_preprocessing(self):
        # dropCols = ["Patient index",
        #             "Annotation meta info",
        #             "Date Of Surgery {1340}",
        dropCols = ["Date Of Surgery {1340}",
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
        # print("Images{:d} shape: ".format(imageId), images.shape)

        return images

    def load_cat(self, imageId):
        imgInfo = self.imageInfo[imageId]
        patientID = int(imgInfo["pstr"].strip("sample"))
        existId = (self.cat_df["Patient index"] == patientID)
        cat = self.cats[existId].iloc[0]
        cat = int(cat > 2)

        return cat

    def get_clinical(self, imageId):
        imgInfo = self.imageInfo[imageId]
        patientID = int(imgInfo["pstr"].strip("sample"))
        existId = (self.cat_df["Patient index"] == patientID)
        clinical_info = self.cat_df[existId]
        assert len(self.cat_df[existId]) == 1
        clinical_info = clinical_info.iloc[0]
        return clinical_info.values[1:]

    def get_cube(self, imageId, size):
        imgInfo = self.imageInfo[imageId]
        data = np.load(imgInfo["imagePath"], allow_pickle=True)
        pos = data["pos"]
        images = data["image"]
#        images = lumTrans(images)
        cube = extract_cube(images, pos, size=size)
        cube = np.array(cube)[np.newaxis, ...]
        return cube

def main():

    Train = True
    use_clinical_features = False
#    rootFolder = "../prepare_for_xinyue/"
#    cat_label_file = "../prepare_for_xinyue/clinical_info.xlsx"
#    load_model_folder = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/model/classification_LUNA16/Resnet18_Adam_lr0.001"

    rootFolder = "/home/c3_server2/Documents/Maoqiang_Python/MICCAI/pretrained_model/Incidental_lung_nodule_data_for_xinyue/Incidental_lung_nodule_data_for_xinyue/"
    cat_label_file = "/home/c3_server2/Documents/Maoqiang_Python/MICCAI/pretrained_model/clinical_info.xlsx"
    load_model_folder = "/home/c3_server2/Documents/Maoqiang_Python/MICCAI/pretrained_model/pretrained_model/Code/model/Resnet18_Adam_lr0.001"

    cube_size = 64
    trainData = LungDataset(rootFolder, cat_label_file=cat_label_file, cube_size=cube_size, train=True, clinical=use_clinical_features)
    trainLoader = DataLoader(trainData, batch_size=1, shuffle=False)

    valData = LungDataset(rootFolder, cat_label_file=cat_label_file, cube_size=cube_size, train=False, clinical=use_clinical_features)
    valLoader = DataLoader(valData, batch_size=1, shuffle=False)
    
    ii=0
    images_train = np.zeros((15,64,64),dtype=float)
    labels_train = np.zeros(15,dtype=float)
    
    for sample_batch in trainLoader:        
        x1, y1 = sample_batch["features"].float(), sample_batch["label"]
        x1 = x1.view(64,64,64)
        x = x1[32]
        x = np.array(x).astype('float64')  #.astype(np.uint8)
        print((x.shape))
        print(ii)
        images_train[ii] = x
        labels_train[ii] = y1
        ii = ii + 1

    ii = 0
    images_test = np.zeros((5,64,64),dtype=float)
    labels_test = np.zeros(5,dtype=float)    
    
    for sample_batch in valLoader:        
        x1, y1 = sample_batch["features"].float(), sample_batch["label"]
        x1 = x1.view(64,64,64)
        x = x1[32]
        x = np.array(x).astype('float64')  #.astype(np.uint8)
        print((x.shape))
        print(ii)
        images_test[ii] = x
        labels_test[ii] = y1
        ii = ii + 1
        
    f = h5py.File('lung_train.h5','w')
    f['x'] = images_train
    f['y'] = labels_train
    f.close() 

    f = h5py.File('lung_test.h5','w')
    f['x'] = images_test
    f['y'] = labels_test
    f.close() 
    
        
        
        
#        ximg = x1[0].reshape(1,64,64).type(torch.FloatTensor).cpu().detach()
#        ximg = np.array(ximg).astype(np.uint8)
#        ximg = np.squeeze(ximg,axis=0)
#        ximg = Image.fromarray(ximg)

    # print ximg
    # ximg.show()   

        
#    dataiter = iter(valLoader)
#    images, labels = dataiter.next()
#    print(labels)
        
if __name__ == '__main__':
    main()
        
        
        
        