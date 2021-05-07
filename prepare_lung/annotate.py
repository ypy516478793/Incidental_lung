from utils.model_utils import plot_bbox
import pandas as pd
import numpy as np
import os

def resample_pos(label, thickness, spacing, new_spacing=[1, 1, 1]):
    spacing = map(float, ([thickness] + list(spacing)))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    resize_factor = resize_factor[::-1]
    label[:3] = np.round(label[:3] * resize_factor)
    label[3] = label[3] * resize_factor[1]
    return label


def annotate(root_dir, annot_file):
    patients = os.listdir(root_dir)
    annot_path = os.path.join(root_dir, annot_file)
    pos_df = pd.read_excel(annot_path, skiprows=1, dtype={"date": str, "size(mm)": str})

    info_path = os.path.join(root_dir, "CTinfo.npz")
    infos = np.load(info_path, allow_pickle=True)["info"]

    pos_df = pos_df[pos_df["z"].notna()]
    # pos_df = pos_df[pos_df["MRN"] > 23043250]
    pos_df = pos_df[pos_df["MRN"] >= 29554987]
    pos_df = pos_df.sort_values(by=["MRN"])

    patient_colname = "patient" if "patient" in pos_df.columns else 'Patient\n Index'
    assert patient_colname in pos_df

    for i in range(len(pos_df)):
        pos_info = pos_df.iloc[i]
        z = pos_info["z"]
        d_ref = pos_info["size(mm)"]
        position = pos_info["position"]
        dstr = pos_info["date"]
        pstr = pos_info[patient_colname]
        match = False
        for j, info in enumerate(infos):
            if info["pstr"] == pstr and info["date"] == dstr:
                match = True
                break
        assert match, "No matches!"
        image_path = info["imagePath"]
        image = np.load(image_path, allow_pickle=True)["image"]
        thickness, spacing = info["sliceThickness"], info["pixelSpacing"]

        new_thickness = new_spacing = 1
        new_z = (z - 1) * thickness / new_thickness

        print("size of nodule for {:s} is: {:s}".format(pstr, d_ref))
        print("position of the nodule: {:s}".format(position))

        ori_label = True
        if ori_label:
            x, y, d = 397, 228.2, 20.2
            new_x, new_y, new_d = (np.array([x, y, d]) - 1) * spacing[0] / new_spacing
            label = np.array([new_x, new_y, new_z, new_d])
        else:
            new_x, new_y, new_d = 153, 235, 11.5
            label = np.array([new_x, new_y, new_z, new_d])
            x, y, d = np.array([new_x, new_y, new_d]) * new_spacing / spacing[0] + 1
        assert spacing[0] == spacing[1]

        print("x: {:.2f}, y: {:.2f}, z: {:.2f}, d: {:.2f}".format(x, y, z, d))
        print("new_x: {:.2f}, new_y: {:.2f}, new_z: {:.2f}, new_d: {:.2f}".format(new_x, new_y, new_z, new_d))
        plot_bbox(image, label, None, True)

        print("=" * 50)



    # for info in infos:
    #     image_path = info["imagePath"]
    #     image = np.load(image_path, allow_pickle=True)["image"]
    #
    #     thickness, spacing = info["sliceThickness"], info["pixelSpacing"]
    #     pstr, dstr = info["pstr"], info["date"]
    #     patient_colname = "patient" if "patient" in pos_df.columns else 'Patient\n Index'
    #     assert patient_colname in pos_df
    #     existId = (pos_df[patient_colname] == pstr) & (pos_df["date"] == dstr)
    #     pos = pos_df[existId][["x", "y", "z", "d"]].values
    #     new_thickness = new_spacing = 1
    #     for p in pos:
    #         z = (p[2] - 1) * thickness / new_thickness
    #
    #         plot_bbox()
    #
    #
    #     pos = np.array([resample_pos(p, thickness, spacing) for p in pos])
    #     pos = pos[:, [2, 1, 0, 3]]
    #
    #     plot_bbox(image, pos, None, True)





def change_root_info(dst_dir):
    file = os.path.join(dst_dir, "CTinfo.npz")
    infos = np.load(file, allow_pickle=True)["info"]
    for info in infos:
        s = info["imagePath"].find("Lung_patient")
        info["imagePath"] = os.path.join(dst_dir, info["imagePath"][s:].replace("\\", "/"))
    print(infos)

    import shutil
    shutil.move(file, os.path.join(dst_dir, "CTinfo_old.npz"))
    np.savez_compressed(file, info=infos)
    print("Save all scan infos to {:s}".format(file))


if __name__ == '__main__':
    root_dir = "/Users/pyuan/Downloads/labeled_TC/"
    annot_file = "Predicted_labels_checklist_Kim_TC.xlsx"
    annotate(root_dir, annot_file)

    # dst_dir = "/Users/pyuan/Downloads/labeled_TC/"
    # change_root_info(dst_dir)