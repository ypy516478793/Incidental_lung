from collections import defaultdict
from natsort import natsorted
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pydicom as dicom
import numpy as np
import scipy.ndimage
import os

def check_fileType(folder, fileType="jpg"):
    '''
    check if there is specific type of file in the folder
    :param folder: folder to be checked
    :param fileType: fileType we are looking for
    :return: a list of files of specified fileType
    '''
    l = [s for s in os.listdir(folder) if s.endswith(fileType)]
    return l

def add_scan(patientID, date, series, imgPath, sliceThickness, pixelSpacing, scanID, **kwargs):
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
    return scanInfo

def read_slices(slices):
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

def load_dicom(imgFolder):

    sliceList = natsorted(os.listdir(imgFolder))
    seriesDict = defaultdict(list)
    for sliceID in sliceList:
        sliceDicom = dicom.read_file(os.path.join(imgFolder, sliceID))
        series = sliceDicom.SeriesDescription
        seriesDict[series].append(sliceDicom)
    patientID = sliceDicom.PatientID
    try:
        date = sliceDicom.ContentDate
    except AttributeError as e:
        print(e)
        date = sliceDicom.StudyDate

    return patientID, date, seriesDict

def resample_label(label, thickness, spacing, new_spacing=[1, 1, 1]):
    spacing = map(float, ([thickness] + list(spacing)))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    resize_factor = resize_factor[::-1]
    label[:3] = np.round(label[:3] * resize_factor)
    label[3] = label[3] * resize_factor[1]

    return label, new_spacing

def resample_image(image, thickness, spacing, new_spacing=[1, 1, 1]):
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

def plot_bbox(images, label, show=True):
    '''
    plot center image with bbox
    :param images: CT scan, shape: (num_slices, h, w) or (h, w)
    :param label: coordinates & diameter (all in pixel space): (x, y, z, d) or (x, y, d)
    :return: None
    '''
    fig, ax = plt.subplots(1)
    if len(label) == 3:
        x, y, d = label
        ax.imshow(images, cmap="gray")
    else:
        x, y, z, d = label
        ax.imshow(images[int(z)], cmap="gray")
    rect = patches.Rectangle((x - d / 2, y - d / 2), d, d, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    if show:
        plt.imshow()
    else:
        plt.savefig()

def extract_cube(images, label, size):
    '''
    extract cube from the CT scan based on the ground truth label.
    :param images: CT scan, shape: (num_slices, h, w)
    :param label: coordinates & diameter (all in pixel space): x, y, z, d
    :param size: size of the cube
    :return: cube centered at nodule's position, shape (num_slices, h, w)
    '''
    x, y, z = label[:3].astype(np.int)
    d = label[-1]
    if size < d:
        print("This nodule is not totally covered in the cube!")
    cube = images[z - size // 2 : z + (size + 1) // 2,
                  y - size // 2 : y + (size + 1) // 2,
                  x - size // 2 : x + (size + 1) // 2]
    return cube

def center_stack(stack, d, rows=5, cols=6, show_every=2, patchType="Circle"):
    '''
    Sample slices from CT scan and show
    :param stack: slices of CT scan
    :param d: diameter of the nodule
    :param rows: rows
    :param cols: cols
    :param show_every: show interval
    :param patchType: Circle or Rectangle
    :return: none
    '''
    fig,ax = plt.subplots(rows,cols,figsize=[12,8])
    num_show = rows*cols
    z, y, x = np.array(stack.shape) // 2
    start_with = z - (num_show // 2 - 1) * show_every
    for i in range(num_show):
        ind = start_with + i*show_every
        ax[int(i/cols),int(i % cols)].set_title('slice %d' % ind)
        ax[int(i/cols),int(i % cols)].imshow(stack[ind],cmap='gray')
        if patchType == "Circle":
            r = np.sqrt(np.max([0, d * d / 4 - (z - ind) * (z - ind)]))
            rect = patches.Circle((x, y), r, linewidth=1, edgecolor='r', facecolor='none')
        else:
            rect = patches.Rectangle((x - d / 2, y - d / 2), d, d, linewidth=1, edgecolor='r', facecolor='none')
        ax[int(i/cols),int(i % cols)].add_patch(rect)
        ax[int(i/cols),int(i % cols)].axis('off')
    plt.show()


# def extract_cube(folder, gt_label_file):
#     '''
#     extract cube from the CT raw data and save it.
#     :param folder: root folder of the data
#     :param gt_label_file: ground truth label csv file
#     :return: None
#     '''
#     label_df = pd.read_csv(os.path.join(folder, gt_label_file))
#     data_folder = os.path.join(folder, "Data")
#     all_patients = [o for o in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, o))]
#
#
#     for i in range(len(label_df)):
#         label = label_df.iloc[i]
#         pId = int(label["patient"][-3:]) - 1
#         assert all_patients[pId].split("-")[0].split("_")[1] == label["patient"]
#         imgFolder = os.path.join(data_folder, all_patients[i], "{:s}_CT_data".format(label["date"]))
#         sliceList = natsorted(os.listdir(imgFolder))
#         patient, date = label["patient"], label["date"]
#         print("\n>>>>>>> Load {:s} at date {:s}".format(patient, date))
#
#         # Distribute all slices to different series
#         seriesDict = defaultdict(list)
#         for sliceID in sliceList:
#             sliceDicom = dicom.read_file(os.path.join(imgFolder, sliceID))
#             series = sliceDicom.SeriesDescription
#             seriesDict[series].append(sliceDicom)
#         patientID = sliceDicom.PatientID
#         assert date == sliceDicom.ContentDate, "Date from dicom does not match that from label."
#         print("PID is: {:s}".format(patientID))
#         print("All series types: ", list(seriesDict.keys()))
#
#         # Load lung series
#         series = label["series"]
#         slices = seriesDict[series]
#         image, sliceThickness, pixelSpacing, scanID = read_slices(slices)
#         imagePath = os.path.join(folder, all_patients[i], "{:s}-{:s}.npz".format(patientID, date))
#         scanInfo = (patientID, date, series, imagePath, sliceThickness, pixelSpacing, scanID)
#         np.savez_compressed(imagePath, image=image, info=scanInfo)
#         print("Save scan to {:s}".format(imagePath))
#
#         return image, scanInfo
#
#
#         # Process only lung scans
#         if len(lungSeries) == 0:
#             print("No lung scans found!")
#             no_CTscans.append(seriesDict)
#         else:
#             # assert len(lungSeries) == 1, "More than 1 lung scans found!"
#             if len(lungSeries) > 1:
#                 print("More than 1 lung scans found!")
#                 id = np.argmin([len(i) for i in lungSeries])
#                 series = lungSeries[id]
#                 matchMoreThanOne.append(lungSeries)
#             else:
#                 series = lungSeries[0]
#             print("Lung series: ", series)
#             slices = seriesDict[series]
#             image, sliceThickness, pixelSpacing, scanID = self.read_slices(slices)
#             imagePath = os.path.join(rootFolder, CTscanId, "{:s}-{:s}.npz".format(patientID, date))
#             scanInfo = self.add_scan(patientID, date, series, imagePath, sliceThickness, pixelSpacing, scanID)
#             np.savez_compressed(imagePath, image=image, info=scanInfo)
#             print("Save scan to {:s}".format(imagePath))
#     CTinfoPath = os.path.join(rootFolder, "CTinfo.npz")
#     np.savez_compressed(CTinfoPath, info=self.imageInfo)
#     print("Save all scan infos to {:s}".format(CTinfoPath))






if __name__ == '__main__':
    root_folder = "I:\Lung_ai"

    # makesense_annots = "labels_lung_ai_20200730113201.csv"
    # create_gt_csv(root_folder, makesense_annots)

    # gt_label = "gt_labels.csv"
    # extract_cube(root_folder, gt_label)