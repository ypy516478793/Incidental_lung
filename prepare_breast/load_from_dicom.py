from tqdm import tqdm

import matplotlib.pyplot as plt
import pydicom as dicom
import pandas as pd
import numpy as np
import os

train = False
root_dir = "/home/cougarnet.uh.edu/pyuan2/Datasets/DDSM/"
data_dir = os.path.join(root_dir, "CBIS-DDSM")


def path_mapping(path):
    str_list = path.split("/")
    scan_dir = os.path.join(data_dir, str_list[0])
    folder = [f for f in os.listdir(scan_dir) if f[-5:] == str_list[1][-5:]][0]
    scan_dir = os.path.join(scan_dir, folder)
    folder = [f for f in os.listdir(scan_dir) if f[-5:] == str_list[2][-5:]][0]
    path = os.path.join(scan_dir, folder, str_list[-1])
    assert os.path.exists(path), "{:s} does not exist!".format(path)
    return path


def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    # slices.sort(key=lambda x: int(x.InstanceNumber))
    slices.sort(key=lambda x: -x.ImagePositionPatient[-1])

    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def main(train):
    train_str = "train" if train else "test"
    if train:
        csv_file = "mass_case_description_train_set.csv"
    else:
        csv_file = "mass_case_description_test_set.csv"

    save_dir = os.path.join(root_dir, "processed_data", "mass", train_str)
    csv_path = os.path.join(root_dir, csv_file)
    df = pd.read_csv(csv_path)

    # Only use the CC view images
    df = df[df["image view"] == "CC"]

    images = []
    masks = []
    cropped_images = []
    dropped = []

    for i in tqdm(range(len(df))):
        info = df.iloc[i]
        image_path = info["image file path"]
        mask_path = info["cropped image file path"]
        cropped_path = info["ROI mask file path"].strip("\n")

        image_path = path_mapping(image_path)
        mask_path = path_mapping(mask_path)
        cropped_path = path_mapping(cropped_path)

        scan = dicom.read_file(image_path)
        image = scan.pixel_array

        scan = dicom.read_file(mask_path)
        mask = scan.pixel_array

        scan = dicom.read_file(cropped_path)
        cropped = scan.pixel_array

        image_id = info["image file path"].split("/")[0]
        save_image_dir = os.path.join(save_dir, image_id)
        os.makedirs(save_image_dir, exist_ok=True)
        save_image_path = os.path.join(save_dir, image_id, "image.npz")
        save_mask_path = os.path.join(save_dir, image_id, "mask.npz")

        if mask.shape == image.shape:
            np.savez_compressed(save_mask_path, mask=mask)
        else:
            if cropped.shape != image.shape:
                print("\nmask shape {:}/cropped shape {:} does not match image shape {:}\nimage id is: {:}".format(
                    mask.shape, cropped.shape, image.shape, image_id))
                dropped.append(image_id)
                continue
            # assert cropped.shape == image.shape, \
            #     "mask shape {:}/cropped shape {:} does not match image shape {:}\n" \
            #     "image id is: {:}".format(mask.shape, cropped.shape, image.shape, image_id)
            np.savez_compressed(save_mask_path, mask=cropped)
        images.append(image)
        np.savez_compressed(save_image_path, image=image)

        # if mask.shape == image.shape:
        #     masks.append(mask)
        #     cropped_images.append(cropped)
        # else:
        #     if cropped.shape != image.shape:
        #         print("\nmask shape {:}/cropped shape {:} does not match image shape {:}\nimage id is: {:}".format(
        #             mask.shape, cropped.shape, image.shape, image_id))
        #         dropped.append(image_id)
        #         continue
        #     # assert cropped.shape == image.shape, \
        #     #     "mask shape {:}/cropped shape {:} does not match image shape {:}\n" \
        #     #     "image id is: {:}".format(mask.shape, cropped.shape, image.shape, image_id)
        #     masks.append(cropped)
        #     cropped_images.append(mask)
        # images.append(image)

        print("")

    print("Total number of {:s} samples: {:d}".format(train_str, len(images)))
    print("Number of dropped samples: {:d}".format(len(dropped)))

    # np.savez_compressed(os.path.join(root_dir, "mass_{:s}.npz".format(train_str)),
    #                     x=images, y=masks)


if __name__ == '__main__':
    main(train)
