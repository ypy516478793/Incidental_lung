from utils.model_utils import extract_cube
from tqdm import tqdm

import pandas as pd
import numpy as np
import os

def prepare_cubes(data_dir, size, label_path=None):

    cat_label_file = "./Methodist_incidental/Methodist_clinical_labels.xlsx"
    cat_df = pd.read_excel(cat_label_file, sheet_name='Sheet1')
    cat_key = [i for i in cat_df.columns if i.startswith("Category Of")][0]
    cat_df["Malignancy"] = (cat_df[cat_key] <= 2).astype(int) # 1=lung cancer, 2=metastatic, 3 = benign nodule,
                                                              # 4= bronchiectasis/pulm sequestration/infection
    cats = dict(cat_df[["Patient", "Malignancy"]].values)


    suffix = "_clean.npz" if label_path is None else ".npz"
    x, y = [], []

    data_ls = []
    for folder in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, folder)):
            for f in os.listdir(os.path.join(data_dir, folder)):
                if f.endswith(suffix):
                    data_ls.append(os.path.join(data_dir, folder, f))

    for data_path in tqdm(data_ls):
        # name = "/".join(data_path.rstrip("_clean.npz").rsplit("/", 2)[1:])

        image = np.load(data_path, allow_pickle=True)["image"]
        if len(image) == 1 and len(image.shape) == 4:
            image = image[0]

        dirname = os.path.dirname(data_path)
        filename = os.path.basename(data_path).rstrip("_clean.npz")

        if label_path is None:
            label_path = os.path.join(dirname, filename + "_label.npz")
            label = np.load(label_path, allow_pickle=True)["label"]
        else:
            pos_df = pd.read_csv(label_path)
            pstr, dstr = filename.split("-")
            patient_colname = "patient" if "patient" in pos_df.columns else 'Patient\n Index'
            assert patient_colname in pos_df
            existId = (pos_df[patient_colname] == pstr) & (pos_df["date"] == int(dstr))
            label = pos_df[existId][["z", "y", "x", "d"]].values

        patient = filename.split("-")[0]
        malignancy = cats[patient]

        # More than one nodule and the patient-level label is malignant
        if malignancy == 1 and len(label) > 1:
            continue

        for i, pos in enumerate(label):
            if pos[-1] > max_nodule_size:
                continue # Not include this nodule if it is larger than the max_nodule_size
            cube = extract_cube(image, pos, size=size)
            x.append(cube)
            y.append(malignancy)

    x = np.expand_dims(np.stack(x), axis=1)
    y = np.array(y)

    save_path = os.path.join(data_dir, "Methodist_3Dcubes_p{:d}.npz".format(size))
    np.savez_compressed(save_path, x=x, y=y)
    print("Save nodule cubes to {:s}".format(save_path))

if __name__ == '__main__':
    # data_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/Methodist_incidental/data_Ben/resampled"
    # label_path = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/Methodist_incidental/data_Ben/resampled/pos_labels_norm.csv"
    # data_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/masked"
    data_dir = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/modeNorm2"
    label_path = None
    size = 64
    max_nodule_size = 60
    prepare_cubes(data_dir, size, label_path=label_path)
