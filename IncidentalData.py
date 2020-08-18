from collections import defaultdict
from natsort import natsorted
from glob import glob
from tqdm import tqdm

from sklearn.cluster import KMeans
from skimage import morphology
from skimage import measure
from skimage.transform import resize

import matplotlib.pyplot as plt
import pydicom as dicom
import numpy as np

import scipy.ndimage
import os


class LungDataset(object):
    def __init__(self, rootFolder):
        self.imageInfo = []
        self._imageIds = []
        self.matches = ["LUNG", "lung"]
        self.load_lung(rootFolder)
        self.prepare()

    def read_slices(self, slices):
        '''
        Read images and other meta_infos from slices list
        :param slices: list of dicom slices
        :return: image in HU and other meta_infos
        '''
        # Sort slices according to the instance number
        slices.sort(key=lambda x: int(x.InstanceNumber))
        image = np.stack([s.pixel_array for s in slices])

        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        intercept = slices[0].RescaleIntercept
        slope = slices[0].RescaleSlope
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)

        # Read some scan properties
        sliceThickness = slices[0].SliceThickness
        pixelSpacing = slices[0].PixelSpacing
        scanID = slices[0].StudyInstanceUID

        return image, sliceThickness, pixelSpacing, scanID

    def add_scan(self, patientID, date, series, imgPath, sliceThickness, pixelSpacing, scanID, **kwargs):
        '''
        Add current scan meta information into global list
        :param: meta informations for current scan
        :return: scan_info (in dictionary)
        '''
        scanInfo = {
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

    def load_from_dicom(self, rootFolder):
        '''
        load image from dicom files
        :param rootFolder: root folder
        :return:
        '''
        no_CTscans = []
        matchMoreThanOne = []
        CTlist = [i for i in os.listdir(rootFolder) if os.path.isdir(os.path.join(rootFolder, i)) \
                                                       and i[:4] == "Lung"]
        CTlist = natsorted(CTlist)
        for CTscanId in CTlist:
            imgFolder = os.path.join(rootFolder, CTscanId, "CT_data")
            sliceList = natsorted(os.listdir(imgFolder))
            # Distribute all slices to different series
            seriesDict = defaultdict(list)
            for sliceID in sliceList:
                sliceDicom = dicom.read_file(os.path.join(imgFolder, sliceID))
                series = sliceDicom.SeriesDescription
                seriesDict[series].append(sliceDicom)
            patientID = sliceDicom.PatientID
            date = sliceDicom.ContentDate
            print("\n>>>>>>> Load patient {:s} at date {:s}".format(patientID, date))
            print("All series types: ", list(seriesDict.keys()))
            lungSeries = [i for i in list(seriesDict.keys()) if np.any([m in i for m in self.matches])]

            # Process only lung scans
            if len(lungSeries) == 0:
                print("No lung scans found!")
                no_CTscans.append(seriesDict)
            else:
                # assert len(lungSeries) == 1, "More than 1 lung scans found!"
                if len(lungSeries) > 1:
                    print("More than 1 lung scans found!")
                    id = np.argmin([len(i) for i in lungSeries])
                    series = lungSeries[id]
                    matchMoreThanOne.append(lungSeries)
                else:
                    series = lungSeries[0]
                print("Lung series: ", series)
                slices = seriesDict[series]
                image, sliceThickness, pixelSpacing, scanID = self.read_slices(slices)
                imagePath = os.path.join(rootFolder, CTscanId, "{:s}-{:s}.npz".format(patientID, date))
                scanInfo = self.add_scan(patientID, date, series, imagePath, sliceThickness, pixelSpacing, scanID)
                np.savez_compressed(imagePath, image=image, info=scanInfo)
                print("Save scan to {:s}".format(imagePath))
        CTinfoPath = os.path.join(rootFolder, "CTinfo.npz")
        np.savez_compressed(CTinfoPath, info=self.imageInfo)
        print("Save all scan infos to {:s}".format(CTinfoPath))

        print("-" * 30 + " CTinfo " + "-" * 30)
        [print(i) for i in self.imageInfo]
        print("-" * 30 + " no_CTscans " + "-" * 30)
        [print(i) for i in [list(i.keys) for i in no_CTscans]]
        print("-" * 30 + " matchMoreThanOne " + "-" * 30)
        [print(i) for i in matchMoreThanOne]

    def load_lung(self, rootFolder):
        try:
            self.imageInfo = np.load(os.path.join(rootFolder, "CTinfo.npz"), allow_pickle=True)["info"]
        except FileNotFoundError:
            self.load_from_dicom(rootFolder)
        # self.load_from_dicom(rootFolder)

    def prepare(self):
        self.num_images = len(self.imageInfo)
        self._imageIds = np.arange(self.num_images)
        self.patient2Image = {"{:s}-{:s}".format(info['patientID'], info['date']): id
                                      for info, id in zip(self.imageInfo, self.imageIds)}

    @property
    def imageIds(self):
        return self._imageIds


    def load_image(self, imageId):
        imgInfo = self.imageInfo[imageId]
        imgPath, thickness, spacing = imgInfo["imagePath"], imgInfo["sliceThickness"], imgInfo["pixelSpacing"]
        image = np.load(imgPath)["image"]
        # plt.imshow(image[10])
        print("Shape before resampling: ", image.shape)
        new_image, new_spacing = resample(image, thickness, spacing)
        print("Shape after resampling: ", new_image.shape)

        return image, new_image

def resample(image, thickness, spacing, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = map(float, ([thickness] + list(spacing)))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

def sample_stack(stack, rows=6, cols=6, start_with=8, show_every=2):
    '''
    Sample slices from CT scan and show
    :param stack: slices of CT scan
    :param rows: rows
    :param cols: cols
    :param start_with: first slice number
    :param show_every: show interval
    :return: none
    '''
    fig,ax = plt.subplots(rows,cols,figsize=[12,8])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/cols),int(i % cols)].set_title('slice %d' % ind)
        ax[int(i/cols),int(i % cols)].imshow(stack[ind],cmap='gray')
        ax[int(i/cols),int(i % cols)].axis('off')
    plt.show()

def plot_hist(img):
    plt.figure()
    plt.hist(img.flatten(), bins=50, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()


# Standardize the pixel values
def make_lungmask(img, display=False):
    row_size = img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    img[img == max] = mean
    img[img == min] = mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))

    labels = measure.label(dilation)  # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[
            2] < col_size / 5 * 4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0  # mask = np.zeros([row_size, col_size], dtype=np.int8)

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask * img, cmap='gray')
        ax[2, 1].axis('off')

        plt.show()
    return mask * img


## --------------- plot 3D --------------- ##
#
# from plotly import __version__
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# from plotly.tools import FigureFactory as FF
# from plotly.graph_objs import *
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#
# def make_mesh(image, threshold=-300, step_size=1):
#     print("Transposing surface")
#     p = image.transpose(2, 1, 0)
#
#     print("Calculating surface")
#     verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)
#     return verts, faces
#
#
# def plotly_3d(verts, faces):
#     x, y, z = zip(*verts)
#
#     print("Drawing")
#
#     # Make the colormap single color since the axes are positional not intensity.
#     #    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
#     colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']
#
#     fig = FF.create_trisurf(x=x,
#                             y=y,
#                             z=z,
#                             plot_edges=False,
#                             colormap=colormap,
#                             simplices=faces,
#                             backgroundcolor='rgb(64, 64, 64)',
#                             title="Interactive Visualization")
#     iplot(fig)
#
#
# def plt_3d(verts, faces):
#     print("Drawing")
#     x, y, z = zip(*verts)
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Fancy indexing: `verts[faces]` to generate a collection of triangles
#     mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
#     face_color = [1, 1, 0.9]
#     mesh.set_facecolor(face_color)
#     ax.add_collection3d(mesh)
#
#     ax.set_xlim(0, max(x))
#     ax.set_ylim(0, max(y))
#     ax.set_zlim(0, max(z))
#     ax.set_axis_bgcolor((0.7, 0.7, 0.7))
#     plt.show()


if __name__ == '__main__':
    rootFolder = "/Users/yuan_pengyu/Downloads/IncidentalLungCTs_sample/"
    lungData = LungDataset(rootFolder)
    image, new_image = lungData.load_image(0)
    img = new_image[100]
    make_lungmask(img, display=True)

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





## --------------- show all images --------------- ##

def plot_no_border(fileName, image, dpi=200, scale=2):
    h, w = image.shape[:2]
    fig, ax = plt.subplots(1, figsize=(w/dpi*scale,h/dpi*scale))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    ax.imshow(image, cmap="gray")
    fig.savefig(fileName, dpi=dpi)
    plt.close(fig)

def makedir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

import matplotlib.patches as patches

def plot_nodule(scan, label, sliceThickness, pixelSpacing, labelFolder):
    '''

    :param scan:
    :param label: shape (z, x, y, d); (sliceId, pixel, pixel, mm)
    :param sliceThickness: CT scan slice thickness (mm/slice)
    :param pixelSpacing: pixel spacing in each CT scan (mm/pixel)
    :return:
    '''
    z, x, y, d = label
    sliceD = d/sliceThickness
    pixelDx = d/pixelSpacing[0]
    pixelDy = d/pixelSpacing[1]

    minz, maxz = z - sliceD, z + sliceD + 0.01
    for i in range(len(scan)):
        if minz <= i and i <= maxz:
            fig, ax = plt.subplots(1)
            ax.imshow(scan[i], cmap="gray")
            rect = patches.Rectangle((x - pixelDx / 2, y - pixelDy / 2), pixelDx, pixelDy, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.savefig(os.path.join(labelFolder, "{:d}_labeled.png".format(i)), dpi=200, bbox_inches="tight")
    plt.show()

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
