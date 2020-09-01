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
    def __init__(self, rootFolder, labeled_only=False, pos_label_file=None, cat_label_file=None, cube_size=64,
                 reload=False, train=True, screen=True):
        self.imageInfo = []
        self._imageIds = []
        self.cube_size = cube_size
        if pos_label_file:
            self.pos_df = pd.read_csv(pos_label_file, dtype={"date": str})
        if cat_label_file:
            self.cat_df = pd.read_excel(cat_label_file, dtype={"MRN": str})
            cat_key = [i for i in self.cat_df.columns if i.startswith("Category Of")][0]
            self.cats = self.cat_df[cat_key]
        self.matches = ["LUNG", "lung"]
        self.load_lung(rootFolder, labeled_only, reload)
        # self.imageInfo = self.imageInfo[:3]
        if screen:
            self.screen()
        self.load_subset(train)
        self.prepare()

    def __len__(self):
        return len(self.imageIds)

    def __getitem__(self, imageId):
        if torch.is_tensor(imageId):
            imageId = imageId.tolist()

        feature = self.get_cube(imageId, self.cube_size)
        # feature = feature[np.newaxis, ...]
        label = self.load_cat(imageId)

        sample = {"features": feature,
                  "label": label}

        return sample

    def add_scan(self, pstr, patientID, date, series, imgPath, sliceThickness, pixelSpacing, scanID, **kwargs):
        '''
        Add current scan meta information into global list
        :param: meta information for current scan
        :return: scan_info (in dictionary)
        '''
        scanInfo = {
            "pstr": pstr,
            "patientID": patientID,
            "scanID": scanID,
            "date": date,
            "series": series,
            "imagePath": imgPath,
            "sliceThickness": sliceThickness,
            "pixelSpacing": pixelSpacing,
        }
        scanInfo.update(kwargs)
        self.imageInfo.append(scanInfo)
        return scanInfo

    def load_from_dicom(self, rootFolder, labeled_only=True):
        '''
        load image from dicom files
        :param rootFolder: root folder of the data
        :return: None
        '''
        no_CTscans = []
        matchMoreThanOne = []
        all_patients = [i for i in os.listdir(rootFolder) if
                        os.path.isdir(os.path.join(rootFolder, i)) and i[:4] == "Lung"]
        all_patients = natsorted(all_patients)

        # Loop over all patients
        for i in range(len(all_patients)):
            patientFolder = os.path.join(rootFolder, all_patients[i])
            all_dates = [d for d in os.listdir(patientFolder) if
                        os.path.isdir(os.path.join(patientFolder, d)) and d[-4:] == "data"]
            # Loop over all dates
            for j in range(len(all_dates)):
                imgFolder = os.path.join(rootFolder, all_patients[i], all_dates[j])
                pstr = all_patients[i].split("-")[0].split("_")[1]
                dstr = all_dates[j].split("_")[0]
                pID = all_patients[i].split("-")[1].split("_")[0]
                imagePath = os.path.join(rootFolder, all_patients[i], "{:s}-{:s}.npz".format(pID, dstr))
                if imagePath in [d["imagePath"] for d in self.imageInfo]:
                    continue
                # find series of only labeled data
                if labeled_only:
                    existId = (self.pos_df["patient"] == pstr) & (self.pos_df["date"] == dstr)
                    if existId.sum() == 0:
                        continue
                    else:
                        series = self.pos_df[existId]["series"].to_numpy()[0]
                        if series == "Lung_Bone+ 50cm":
                            series = series.replace("_", "/")
                        print("\n>>>>>>> Start to load {:s} at date {:s}".format(pstr, dstr))

                # Distribute all slices to different series
                patientID, dateDicom, seriesDict = load_dicom(imgFolder)
                assert patientID == pID, "PatientID does not match!!"
                assert dateDicom == dstr, "Date does not match!!"
                print("All series types: ", list(seriesDict.keys()))

                # find series of unlabeled data based on the matches pattern (self.matches)
                if not labeled_only:
                    lungSeries = [i for i in list(seriesDict.keys()) if np.any([m in i for m in self.matches])]
                    if len(lungSeries) == 0:
                        print("No lung scans found!")
                        no_CTscans.append(seriesDict)
                    else:
                        if len(lungSeries) > 1:
                            print("More than 1 lung scans found!")
                            id = np.argmin([len(i) for i in lungSeries])
                            series = lungSeries[id]
                            matchMoreThanOne.append(lungSeries)
                        else:
                            series = lungSeries[0]
                        print("Lung series: ", series)

                # Load and save lung series
                slices = seriesDict[series]
                image, sliceThickness, pixelSpacing, scanID = read_slices(slices)
                # imagePath = os.path.join(rootFolder, all_patients[i], "{:s}-{:s}.npz".format(patientID, dateDicom))
                scanInfo = self.add_scan(pstr, patientID, dateDicom, series, imagePath,
                                         sliceThickness, pixelSpacing, scanID)
                new_image, new_spacing = resample_image(image, sliceThickness, pixelSpacing)
                np.savez_compressed(imagePath, image=new_image, info=scanInfo)
                print("Save scan to {:s}".format(imagePath))

                print("\nFinish loading patient {:s} at date {:s} <<<<<<<".format(patientID, dateDicom))

                CTinfoPath = os.path.join(rootFolder, "CTinfo.npz")
                np.savez_compressed(CTinfoPath, info=self.imageInfo)
                print("Save all scan infos to {:s}".format(CTinfoPath))

        print("-" * 30 + " CTinfo " + "-" * 30)
        [print(i) for i in self.imageInfo]

    def load_lung(self, rootFolder, labeled_only, reload=False):
        if reload:
            self.imageInfo = np.load(os.path.join(rootFolder, "CTinfo.npz"), allow_pickle=True)["info"].tolist()
            self.load_from_dicom(rootFolder, labeled_only=labeled_only)
        else:
            try:
                self.imageInfo = np.load(os.path.join(rootFolder, "CTinfo.npz"), allow_pickle=True)["info"]
            except FileNotFoundError:
                self.load_from_dicom(rootFolder, labeled_only=labeled_only)

    def load_subset(self, train):
        trainInfo, valInfo = train_test_split(self.imageInfo, random_state=42)
        # trainInfo, valInfo = train_test_split(self.imageInfo)
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
            if len(pos) > 1:
                mask[imageId] = False
        self.imageInfo = self.imageInfo[mask]


    def load_image(self, imageId):
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
                np.savez_compressed(cubePath, image=cube, info=imgInfo, pos=p)
                print("Save scan cube to {:s}".format(cubePath))
            cubes.append(cube)
        cubes = np.array(cubes)

        return cubes


if __name__ == '__main__':
    # rootFolder = "/Users/yuan_pengyu/Downloads/IncidentalLungCTs_sample/"
    rootFolder = "data/"
    pos_label_file = "data/pos_labels.csv"
    cat_label_file = "data/Lung Nodule Clinical Data_Min Kim (No name).xlsx"
    cube_size = 64
    lungData = LungDataset(rootFolder, labeled_only=True, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
                           cube_size=cube_size, reload=False)

    # for i in range(len(lungData.imageInfo)):
    #     s = lungData.imageInfo[i]["imagePath"]
    #     lungData.imageInfo[i]["imagePath"] = s.replace("\\", "/").replace("I:/Lung_ai/Data/", "../data/")

    print()
    # image, new_image = lungData.load_image(0)
    # img = new_image[100]
    # make_lungmask(img, display=True)

    from prepare import show_nodules
    crop_size = 64
    show_nodules(lungData, crop_size)



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

