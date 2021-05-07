from utils.model_utils import read_slices, load_dicom, resample_image
from natsort import natsorted
import pandas as pd
import numpy as np
import os


def add_scan(pstr, patientID, date, series, imgPath, sliceThickness, pixelSpacing, scanID, **kwargs):
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
    return scanInfo


def load_from_dicom(root_dir, imageInfo, pos_df, labeled=True):
    '''
    load image from dicom files
    :param root_dir: root folder of the data
    :param labeled: {True: only labeled samples, False: only unlabeled samples, None: all samples}
    :return: None
    '''
    matches = ["LUNG", "lung"]
    no_CTscans = []
    matchMoreThanOne = []
    all_patients = [i for i in os.listdir(root_dir) if
                    os.path.isdir(os.path.join(root_dir, i)) and i[:4] == "Lung"]
    all_patients = natsorted(all_patients)

    labeled_str = "labeled" if labeled else "unlabeled"
    save_dir = os.path.join(os.path.dirname(root_dir), "processed_data", labeled_str)

    # Loop over all patients
    for i in range(len(all_patients)):
        patientFolder = os.path.join(root_dir, all_patients[i])
        all_dates = [d for d in os.listdir(patientFolder) if
                     os.path.isdir(os.path.join(patientFolder, d)) and d[-4:] == "data"]
        # Loop over all dates
        for j in range(len(all_dates)):
            imgFolder = os.path.join(root_dir, all_patients[i], all_dates[j])
            pstr = all_patients[i].split("-")[0].split("_")[1]
            dstr = all_dates[j].split("_")[0]
            pID = all_patients[i].split("-")[1].split("_")[0]
            save_patient_dir = os.path.join(save_dir, all_patients[i])
            os.makedirs(save_patient_dir, exist_ok=True)
            image_path = os.path.join(save_patient_dir, "{:s}-{:s}.npz".format(pID, dstr))
            if image_path in [d["imagePath"] for d in imageInfo]:
                continue
            # find series of only labeled data
            has_label = False
            if pos_df is not None:
                existId = (pos_df["patient"] == pstr) & (pos_df["date"] == dstr)
                if existId.sum() != 0:
                    has_label = True
                    series = pos_df[existId]["series"].to_numpy()[0]
                    if series == "Lung_Bone+ 50cm" or series == "LUNG_BONE PLUS 50cm":
                        series = series.replace("_", "/")
                    # print("\n>>>>>>> Start to load {:s} at date {:s}".format(pstr, dstr))
            if labeled is not None:
                if labeled and not has_label:
                    continue
                if not labeled and has_label:
                    continue

            # Distribute all slices to different series
            patientID, dateDicom, seriesDict = load_dicom(imgFolder)
            assert patientID == pID, "PatientID does not match!!"
            assert dateDicom == dstr, "Date does not match!!"
            print("\n>>>>>>> Start to load {:s} at date {:s}".format(pstr, dstr))
            print("All series types: ", list(seriesDict.keys()))

            if not has_label:
                lungSeries = [i for i in list(seriesDict.keys()) if np.any([m in i for m in matches])]
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


            # if labeled:
            #     existId = (pos_df["patient"] == pstr) & (pos_df["date"] == dstr)
            #     if existId.sum() == 0:
            #         continue
            #     else:
            #         series = pos_df[existId]["series"].to_numpy()[0]
            #         if series == "Lung_Bone+ 50cm" or series == "LUNG_BONE PLUS 50cm":
            #             series = series.replace("_", "/")
            #         print("\n>>>>>>> Start to load {:s} at date {:s}".format(pstr, dstr))
            # 
            # # Distribute all slices to different series
            # patientID, dateDicom, seriesDict = load_dicom(imgFolder)
            # assert patientID == pID, "PatientID does not match!!"
            # assert dateDicom == dstr, "Date does not match!!"
            # print("All series types: ", list(seriesDict.keys()))
            # 
            # # find series of unlabeled data based on the matches pattern (matches)
            # if not labeled:
            #     lungSeries = [i for i in list(seriesDict.keys()) if np.any([m in i for m in matches])]
            #     if len(lungSeries) == 0:
            #         print("No lung scans found!")
            #         no_CTscans.append(seriesDict)
            #     else:
            #         if len(lungSeries) > 1:
            #             print("More than 1 lung scans found!")
            #             id = np.argmin([len(i) for i in lungSeries])
            #             series = lungSeries[id]
            #             matchMoreThanOne.append(lungSeries)
            #         else:
            #             series = lungSeries[0]
            #         print("Lung series: ", series)

            # Load and save lung series
            slices = seriesDict[series]
            image, sliceThickness, pixelSpacing, scanID = read_slices(slices)
            # image_path = os.path.join(root_dir, all_patients[i], "{:s}-{:s}.npz".format(patientID, dateDicom))
            scanInfo = add_scan(pstr, patientID, dateDicom, series, image_path,
                                sliceThickness, pixelSpacing, scanID)
            imageInfo.append(scanInfo)
            new_image, new_spacing = resample_image(image, sliceThickness, pixelSpacing)
            np.savez_compressed(image_path, image=new_image, info=scanInfo)
            print("Save scan to {:s}".format(image_path))

            print("\nFinish loading patient {:s} at date {:s} <<<<<<<".format(patientID, dateDicom))

            CTinfoPath = os.path.join(save_dir, "CTinfo.npz")
            np.savez_compressed(CTinfoPath, info=imageInfo)
            print("Save all scan infos to {:s}".format(CTinfoPath))

    print("-" * 30 + " CTinfo " + "-" * 30)
    [print(i) for i in imageInfo]

def change_path(info_dir):
    info_path = os.path.join(info_dir, "CTinfo.npz")
    imageInfo = np.load(info_path, allow_pickle=True)["info"]
    for info in imageInfo:
        old_image_path = info["imagePath"]
        image_path = old_image_path.replace("\\", "/")
        s = image_path.find("Lung_patient")
        new_image_path = os.path.join(root_dir, image_path[s:])
        if new_image_path == old_image_path:
            print("Do not need to change the image path.")
            return imageInfo
        assert os.path.exists(new_image_path), "{:s} does not exist!".format(new_image_path)
        info["imagePath"] = new_image_path
    os.rename(info_path, os.path.join(info_dir, "CTinfo_old.npz"))
    np.savez_compressed(info_path, info=imageInfo)
    return imageInfo


def check_num_nodules(file = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/Dataset_details_0916.xlsx"):
    df = pd.read_excel(file)
    from ast import literal_eval
    ids = [id for id, i in enumerate(df.iloc[:, 3]) if
           i is not np.nan and isinstance(literal_eval(i), dict) and list(literal_eval(i).values())[0] > 1]
    return ids

def move_npz(save_dir, data_folder):
    from tqdm import tqdm
    from shutil import copyfile
    file = os.path.join(save_dir, "move_folder.csv")
    with open(file, "r") as f:
        lines = f.readlines()
        for l in tqdm(lines):
            src, dst = l.split(",")
            src, dst = src.strip(), dst.strip()
            src = os.path.join(data_folder, *src.replace("\\", "/").split("/")[-2:])
            dst = os.path.join(save_dir, *dst.replace("\\", "/").split("/")[-2:])
            if not os.path.exists(dst):
                assert os.path.exists(src), "src {:} does not exist.".format(src)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                copyfile(src.strip(), dst.strip())


if __name__ == '__main__':
    
    # root_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_mamta/"
    # raw_data_dir = os.path.join(root_dir, "raw_data")
    # # pos_label_file = os.path.join(raw_data_dir, "pos_labels.csv")
    # # cat_label_file = "data/Lung Nodule Clinical Data_Min Kim - Added Variables 10-2-2020.xlsx"
    # # pos_df = pd.read_csv(pos_label_file, dtype={"date": str})
    # pos_df = None
    # imageInfo = []
    #
    # imageInfo = load_from_dicom(raw_data_dir, imageInfo, pos_df, labeled=False)

    ## ----- Change imagePath in imageInfo ----- ##
    # root_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/labeled"
    # imageInfo = change_path(root_dir)
    # print("")

    ## ----- Combine existing npz with additional npz ----- ##
    root_dir = "/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_Ben/labeled"
    data_folder = "/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_kim/labeled"
    imageInfo = move_npz(root_dir, data_folder)
    print("")