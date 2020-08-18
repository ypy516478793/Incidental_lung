from shutil import copyfile
import pandas as pd
import numpy as np
import os

def organize_screenshots(folder, report_file):
    '''
    Copy all screenshots to one folder
    :param folder: root folder of the data
    :param report_file: full clinical report file name
    :return: None
    '''
    report = pd.read_excel(os.path.join(root_folder, report_file))
    meta_info = report["Annotation meta info"]
    data_folder = os.path.join(folder, "Data")
    dst_folder = os.path.join(folder, "Screenshots")
    os.makedirs(dst_folder, exist_ok=True)
    all_patients = [o for o in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder,o))]
    for patient in all_patients:
        pstr = patient.split("-")[0].split("_")[1]
        pId = int(pstr[-3:]) - 1
        try:
            dstr = meta_info[pId].split("-")[0]
        except:
            dstr = "noDate"
        patient_folder = os.path.join(data_folder, patient)
        images = [i for i in os.listdir(patient_folder) if i.endswith('jpg')]
        for img in images:
            img_file = os.path.join(patient_folder, img)
            dst_file = os.path.join(dst_folder, "{:s}-{:s}-{:s}".format(pstr, dstr, img))
            copyfile(img_file, dst_file)

def organize_npz(folder):
    data_folder = os.path.join(folder, "Data")
    dst_folder = os.path.join(folder, "processed_data")
    os.makedirs(dst_folder, exist_ok=True)
    all_patients = [o for o in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder,o))]
    for patient in all_patients:
        pstr = patient.split("-")[0].split("_")[1]
        patient_folder = os.path.join(data_folder, patient)
        ct_npz = [i for i in os.listdir(patient_folder) if i.endswith('npz') and "_" not in i]
        for npz in ct_npz:
            npz_file = os.path.join(patient_folder, npz)
            dst_file = os.path.join(dst_folder, "{:s}-{:s}".format(pstr, npz))
            copyfile(npz_file, dst_file)

def create_gt_csv(folder, annot_file):
    '''
    create the ground truth csv file based on the annotation information.
    :param folder: root folder of the data
    :param annot_file: makesense exported annotation csv file
    :return: None
    '''
    annotations = pd.read_csv(os.path.join(folder, annot_file),
                              names=["label", "x", "y", "w", "h", "imgName", "W", "H"])
    labels = []
    for i in range(len(annotations)):
        patient_obj = annotations.iloc[i]
        annot_list = patient_obj["imgName"].split("-", 2)
        pstr, dstr = annot_list[:2]
        dstr = dstr[4:] + dstr[:4]
        imgName_list = annot_list[2].split(".")
        Series = imgName_list[1]
        z, Z = [int(j) for j in imgName_list[-2][4:].split("_")]
        x = patient_obj["x"] + patient_obj["w"] / 2
        y = patient_obj["y"] + patient_obj["h"] / 2
        d = np.sqrt(np.power(patient_obj["w"], 2) + np.power(patient_obj["h"], 2))
        labels.append((pstr, dstr, Series, x, y, z, d))
    columns = ["patient", "date", "series", "x", "y", "z", "d"]
    label_df = pd.DataFrame(labels, columns=columns)
    label_df.to_csv(os.path.join(folder, "gt_labels.csv"), index=False)


if __name__ == '__main__':
    root_folder = "I:\Lung_ai"

    # Kim_report = "Lung Nodule Clinical Data_Min Kim (Autosaved).xlsx"
    # organize_screenshots(root_folder, Kim_report)

    # makesense_annots = "labels_lung_ai_20200805015042.csv"
    # create_gt_csv(root_folder, makesense_annots)

    # gt_label = "gt_labels.csv"
    # extract_cube(root_folder, gt_label)

    organize_npz(root_folder)