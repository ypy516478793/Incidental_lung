from utils.model_utils import extract_cube, resample_pos, lumTrans, collate, plot_bbox, center_stack
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

import os

class LungDataset(Dataset):
    def __init__(self, rootFolder, pos_label_file=None, cat_label_file=None, cube_size=64,
                 train=None, screen=True, clinical=False):
        self.imageInfo = []
        self._imageIds = []
        self.cube_size = cube_size
        if pos_label_file:
            self.pos_df = pd.read_csv(pos_label_file, dtype={"date": str})
        if cat_label_file:
            self.cat_df = pd.read_excel(cat_label_file, dtype={"MRN": str}, sheet_name='Sheet1')
            cat_key = [i for i in self.cat_df.columns if i.startswith("Category Of")][0]
            self.cats = self.cat_df[cat_key]
            if clinical:
                try:
                    self.additional_df = pd.read_excel(cat_label_file, dtype={"MRN": str}, sheet_name='Sheet2')
                    self.clinical_preprocessing()
                except:
                    print("No clinical information availabel")
                    clinical = False
        self.load_clinical = clinical
        self.imageInfo = np.load(os.path.join(rootFolder, "CTinfo.npz"), allow_pickle=True)["info"]
        self.imageInfo = np.array(self.imageInfo)
        if screen:
            self.screen()
        if train is not None:
            self.load_subset(train)
        self.prepare()

    def __len__(self):
        return len(self.imageIds)

    def __getitem__(self, imageId):
        if torch.is_tensor(imageId):
            imageId = imageId.tolist()

        image = self.load_image(imageId)
        pos = self.load_pos(imageId)
        cubes = self.get_cube(imageId, self.cube_size)
        label = self.load_cat(imageId)

        sample = {"image": image,
                  "pos": pos,
                  "cubes": cubes,
                  "label": label}

        if self.load_clinical:
            clinical = self.get_clinical(imageId).astype(np.float32)
            sample = sample.update({"clinical": clinical})

        return sample

    def load_subset(self, train):
        trainInfo, valInfo = train_test_split(self.imageInfo, random_state=42)
        self.imageInfo = trainInfo if train else valInfo

    def prepare(self):
        self.num_images = len(self.imageInfo)
        self._imageIds = np.arange(self.num_images)
        self.patient2Image = {"{:s}-{:s}".format(info['patientID'], info['date']): id
                                      for info, id in zip(self.imageInfo, self.imageIds)}
        self.patient2Image.update({"{:s}-{:s}".format(info['pstr'], info['date']): id
                                  for info, id in zip(self.imageInfo, self.imageIds)})
    @property
    def imageIds(self):
        return self._imageIds

    def screen(self):
        '''
        Only maintain the cases with multiple nodules
        '''
        num_images = len(self.imageInfo)
        mask = np.ones(num_images, dtype=bool)
        for imageId in range(num_images):
            pos = self.load_pos(imageId)
            if len(pos) <= 1:
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
        '''
        Load CT scan images, shape == (1, nz, nh, nw)
        '''
        imgInfo = self.imageInfo[imageId]
        imgPath, thickness, spacing = imgInfo["imagePath"], imgInfo["sliceThickness"], imgInfo["pixelSpacing"]
        images = np.load(imgPath)["image"]
        images = lumTrans(images)
        # if mask:
        #     masked_images = []
        #     for img in images:
        #         masked_images.append(make_lungmask(img))
        #     masked_images = np.stack(masked_images)
        # plt.imshow(images[10])
        print("Images{:d} shape: ".format(imageId), images.shape)

        return images

    def load_pos(self, imageId):
        '''
        Load position of nodules, shape == (# of nodules, 4); dim[-1] -> (x, y, z, d)
        '''
        imgInfo = self.imageInfo[imageId]
        thickness, spacing = imgInfo["sliceThickness"], imgInfo["pixelSpacing"]
        pstr = imgInfo["pstr"]
        dstr = imgInfo["date"]
        existId = (self.pos_df["patient"] == pstr) & (self.pos_df["date"] == dstr)
        pos = self.pos_df[existId][["x", "y", "z", "d"]].values
        pos = np.array([resample_pos(p, thickness, spacing) for p in pos])

        return pos

    def load_cat(self, imageId):
        '''
        Load categorical label, 0: malignent, 1: benign
        '''
        imgInfo = self.imageInfo[imageId]
        patientID = imgInfo["patientID"]
        existId = (self.cat_df["MRN"].str.zfill(9) == patientID)
        cat = self.cats[existId].iloc[0]
        cat = int(cat > 2)

        return cat

    def get_clinical(self, imageId):
        '''
        Load clinical features
        '''
        imgInfo = self.imageInfo[imageId]
        pid = int(imgInfo["pstr"][-3:])
        assert self.cat_df.iloc[pid-1]["MRN"].zfill(9) == imgInfo["patientID"]
        clinical_info = self.cat_df.iloc[pid-1]
        return clinical_info.values[1:]

    def get_cube(self, imageId, size):
        '''
        Crop a cube centered at the each nodule
        '''
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
    rootFolder = "data_king/labeled/"
    pos_label_file = "../data/pos_labels.csv"
    # cat_label_file = "data/Lung Nodule Clinical Data_Min Kim - Added Variables 10-2-2020.xlsx"
    cat_label_file = "../data/Lung Nodule Clinical Data_Min Kim (No name).xlsx"
    cube_size = 64
    lungData = LungDataset(rootFolder, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
                           cube_size=cube_size, train=None, screen=True, clinical=True)


    # Load one CT scan and plot the nodule
    imageId = 0
    image = lungData.load_image(imageId)
    pos = lungData.load_pos(imageId)
    label = lungData.load_cat(imageId)
    plot_bbox(image, pos[0], None, show=True)



    # Load batch data through dataloader
    from torch.utils.data import DataLoader
    dataLoader = DataLoader(lungData, batch_size=2, drop_last=False, collate_fn=collate)
    for sample in dataLoader:
        image, pos, cubes, label = sample["image"], sample["pos"], sample["cubes"], sample["label"]
        plt.imshow(cubes[0][0, 32], cmap="gray")
        plt.title("central slice of the cube")
        plt.show()

        imageId = 0
        noduleId = 0
        d = pos[imageId][noduleId, -1]
        center_stack(cubes[imageId][noduleId], d, None, show=True)

        print("")



