from collections import defaultdict
from natsort import natsorted
from tqdm import tqdm
from utils import check_fileType, read_slices, load_dicom, extract_cube, resample_image, resample_pos, make_lungmask, lumTrans
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pydicom as dicom
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
            self.additional_df = pd.read_excel(cat_label_file, dtype={"MRN": str}, sheet_name='Sheet2')
            cat_key = [i for i in self.cat_df.columns if i.startswith("Category Of")][0]
            self.cats = self.cat_df[cat_key]
        self.clinical_preprocessing()
        self.load_clinical = clinical
        self.imageInfo = np.load(os.path.join(rootFolder, "CTinfo.npz"), allow_pickle=True)["info"]
        self.imageInfo = np.array(self.imageInfo)
        # self.imageInfo = self.imageInfo[:3]
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
        # if len(feature) > 1:
        #     assert label == 1, "Must be benign cases!"
        #     feature = feature[[np.random.randint(len(feature))]]

        # if self.load_clinical:
        #     feature = (feature, feature_clinical)

        sample = {"image": image,
                  "pos": pos,
                  "cubes": cubes,
                  "label": label}

        if self.load_clinical:
            clinical = self.get_clinical(imageId).astype(np.float32)
            sample = sample.update({"clinical": clinical})

        return sample

    # def add_scan(self, pstr, patientID, date, series, imgPath, sliceThickness, pixelSpacing, scanID, **kwargs):
    #     '''
    #     Add current scan meta information into global list
    #     :param: meta information for current scan
    #     :return: scan_info (in dictionary)
    #     '''
    #     scanInfo = {
    #         "pstr": pstr,
    #         "patientID": patientID,
    #         "scanID": scanID,
    #         "date": date,
    #         "series": series,
    #         "imagePath": imgPath,
    #         "sliceThickness": sliceThickness,
    #         "pixelSpacing": pixelSpacing,
    #     }
    #     scanInfo.update(kwargs)
    #     self.imageInfo.append(scanInfo)
    #     return scanInfo
    #
    # def load_from_dicom(self, rootFolder, labeled_only=True):
    #     '''
    #     load image from dicom files
    #     :param rootFolder: root folder of the data
    #     :return: None
    #     '''
    #     no_CTscans = []
    #     matchMoreThanOne = []
    #     all_patients = [i for i in os.listdir(rootFolder) if
    #                     os.path.isdir(os.path.join(rootFolder, i)) and i[:4] == "Lung"]
    #     all_patients = natsorted(all_patients)
    #
    #     # Loop over all patients
    #     for i in range(len(all_patients)):
    #         patientFolder = os.path.join(rootFolder, all_patients[i])
    #         all_dates = [d for d in os.listdir(patientFolder) if
    #                     os.path.isdir(os.path.join(patientFolder, d)) and d[-4:] == "data"]
    #         # Loop over all dates
    #         for j in range(len(all_dates)):
    #             imgFolder = os.path.join(rootFolder, all_patients[i], all_dates[j])
    #             pstr = all_patients[i].split("-")[0].split("_")[1]
    #             dstr = all_dates[j].split("_")[0]
    #             pID = all_patients[i].split("-")[1].split("_")[0]
    #             imagePath = os.path.join(rootFolder, all_patients[i], "{:s}-{:s}.npz".format(pID, dstr))
    #             if imagePath in [d["imagePath"] for d in self.imageInfo]:
    #                 continue
    #             # find series of only labeled data
    #             if labeled_only:
    #                 existId = (self.pos_df["patient"] == pstr) & (self.pos_df["date"] == dstr)
    #                 if existId.sum() == 0:
    #                     continue
    #                 else:
    #                     series = self.pos_df[existId]["series"].to_numpy()[0]
    #                     if series == "Lung_Bone+ 50cm" or series == "LUNG_BONE PLUS 50cm":
    #                         series = series.replace("_", "/")
    #                     print("\n>>>>>>> Start to load {:s} at date {:s}".format(pstr, dstr))
    #
    #             # Distribute all slices to different series
    #             patientID, dateDicom, seriesDict = load_dicom(imgFolder)
    #             assert patientID == pID, "PatientID does not match!!"
    #             assert dateDicom == dstr, "Date does not match!!"
    #             print("All series types: ", list(seriesDict.keys()))
    #
    #             # find series of unlabeled data based on the matches pattern (self.matches)
    #             if not labeled_only:
    #                 lungSeries = [i for i in list(seriesDict.keys()) if np.any([m in i for m in self.matches])]
    #                 if len(lungSeries) == 0:
    #                     print("No lung scans found!")
    #                     no_CTscans.append(seriesDict)
    #                 else:
    #                     if len(lungSeries) > 1:
    #                         print("More than 1 lung scans found!")
    #                         id = np.argmin([len(i) for i in lungSeries])
    #                         series = lungSeries[id]
    #                         matchMoreThanOne.append(lungSeries)
    #                     else:
    #                         series = lungSeries[0]
    #                     print("Lung series: ", series)
    #
    #             # Load and save lung series
    #             slices = seriesDict[series]
    #             image, sliceThickness, pixelSpacing, scanID = read_slices(slices)
    #             # imagePath = os.path.join(rootFolder, all_patients[i], "{:s}-{:s}.npz".format(patientID, dateDicom))
    #             scanInfo = self.add_scan(pstr, patientID, dateDicom, series, imagePath,
    #                                      sliceThickness, pixelSpacing, scanID)
    #             new_image, new_spacing = resample_image(image, sliceThickness, pixelSpacing)
    #             np.savez_compressed(imagePath, image=new_image, info=scanInfo)
    #             print("Save scan to {:s}".format(imagePath))
    #
    #             print("\nFinish loading patient {:s} at date {:s} <<<<<<<".format(patientID, dateDicom))
    #
    #             CTinfoPath = os.path.join(rootFolder, "CTinfo.npz")
    #             np.savez_compressed(CTinfoPath, info=self.imageInfo)
    #             print("Save all scan infos to {:s}".format(CTinfoPath))
    #
    #     print("-" * 30 + " CTinfo " + "-" * 30)
    #     [print(i) for i in self.imageInfo]

    # def load_lung(self, rootFolder, labeled_only, reload=False):
    #     if reload:
    #         self.imageInfo = np.load(os.path.join(rootFolder, "CTinfo.npz"), allow_pickle=True)["info"].tolist()
    #         self.load_from_dicom(rootFolder, labeled_only=labeled_only)
    #     else:
    #         try:
    #             self.imageInfo = np.load(os.path.join(rootFolder, "CTinfo.npz"), allow_pickle=True)["info"]
    #         except FileNotFoundError:
    #             self.load_from_dicom(rootFolder, labeled_only=labeled_only)
    #     self.imageInfo = np.array(self.imageInfo)

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
        num_images = len(self.imageInfo)
        mask = np.ones(num_images, dtype=bool)
        for imageId in range(num_images):
            pos = self.load_pos(imageId)
            cat = self.load_cat(imageId)
            if len(pos) > 1:
            # if len(pos) > 1 and cat == 0:
            # if len(pos) <= 1:
                mask[imageId] = False
        self.imageInfo = self.imageInfo[mask]

    # def screen(self):
    #     num_images = len(self.imageInfo)
    #     mask = np.ones(num_images, dtype=bool)
    #     for imageId in range(num_images):
    #         pos = self.load_pos(imageId)
    #         cat = self.load_cat(imageId)
    #         if len(pos) > 1 and cat == 0:
    #         # if len(pos) > 1:
    #             mask[imageId] = False
    #     self.imageInfo = self.imageInfo[mask]


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
        thickness, spacing = imgInfo["sliceThickness"], imgInfo["pixelSpacing"]
        pstr = imgInfo["pstr"]
        dstr = imgInfo["date"]
        existId = (self.pos_df["patient"] == pstr) & (self.pos_df["date"] == dstr)
        pos = self.pos_df[existId][["x", "y", "z", "d"]].values
        pos = np.array([resample_pos(p, thickness, spacing) for p in pos])

        return pos

    def load_cat(self, imageId):
        imgInfo = self.imageInfo[imageId]
        patientID = imgInfo["patientID"]
        existId = (self.cat_df["MRN"].str.zfill(9) == patientID)
        cat = self.cats[existId].iloc[0]
        cat = int(cat > 2)

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
    # rootFolder = "/Users/yuan_pengyu/Downloads/IncidentalLungCTs_sample/"
    # rootFolder = "data/"
    rootFolder = "data_king/labeled/"
    # rootFolder = "data_king/unlabeled/"
    pos_label_file = "data/pos_labels.csv"
    cat_label_file = "data/Lung Nodule Clinical Data_Min Kim - Added Variables 10-2-2020.xlsx"
    cube_size = 64
    lungData = LungDataset(rootFolder, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
                           cube_size=cube_size, train=None, screen=True, clinical=False)
    # image, new_image = lungData.load_image(0)
    # img = new_image[100]
    # make_lungmask(img, display=True)

    # from prepare_lung import show_nodules
    # crop_size = 64
    # show_nodules(lungData, crop_size)
    #
    # lungData_test = LungDataset(rootFolder, labeled_only=True, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
    #                        cube_size=cube_size, reload=False, train=False)
    # crop_size = 64
    # show_nodules(lungData_test, crop_size, train=False)



    # saveFolder = "./data/"
    # for id in tqdm(lungData.imageIds):
    #     image, new_image = lungData.load_image(id)
    #     masked_lung = []
    #     for img in new_image:
    #         masked_lung.append(make_lungmask(img))
    #     masked_lung = np.array(masked_lung)
    #     fileName = "CT_scan_{:d}".format(id)
    #     np.save(os.path.join(saveFolder, fileName + "_clean.npy"), masked_lung)
    #     np.save(os.path.join(saveFolder, fileName + "_label.npy"), np.array([]))
    #     print("Save data_{:d} to {:s}".format(id, os.path.join(saveFolder, fileName + "_clean.npy")))

    from torch.utils.data import DataLoader
    from utils import collate
    dataLoader = DataLoader(lungData, batch_size=2, drop_last=False, collate_fn=collate)
    for sample in dataLoader:
        image, cubes, label = sample["image"], sample["cubes"], sample["label"]
        print("")

    print("")



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

