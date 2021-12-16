from tqdm import tqdm
import SimpleITK as sitk
import pandas as pd
import numpy as np
import os

# class NoduleFinding(object):
#     """
#     Represents a nodule
#     """
#
#     def __init__(self, noduleid=None, coordX=None, coordY=None, coordZ=None, coordType="World",
#                  CADprobability=None, noduleType=None, diameter=None, state=None, seriesInstanceUID=None):
#         # set the variables and convert them to the correct type
#         self.id = noduleid
#         self.coordX = coordX
#         self.coordY = coordY
#         self.coordZ = coordZ
#         self.coordType = coordType
#         self.CADprobability = CADprobability
#         self.noduleType = noduleType
#         self.diameter_mm = diameter
#         self.state = state
#         self.candidateID = None
#         self.seriesuid = seriesInstanceUID

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith("TransformMatrix")][0]
        transformM = np.array(line.split(" = ")[1].split(" ")).astype("float")
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing,isflip

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def getvoxcrd(srsid, candidates):
    sliceim, origin, spacing, isflip = load_itk_image(os.path.join(srsid_path_dict[srsid]))
    voxcrdlist = []
    for lunaant in candidates:
        voxcrd = worldToVoxelCoord(lunaant[:3][::-1], origin, spacing)
        voxcrd[-1] = sliceim.shape[0] - voxcrd[0]
        voxcrdlist.append(voxcrd)
    return voxcrdlist

class Nodule(object):
    def __init__(self, noduleId=None, coordWorld=None, coordVoxel=None, diameter=None, label=None):
        """ label: {1: nodule, 0: non-nodule}"""
        self.id = noduleId
        self.coordWorld = coordWorld
        self.coordVoxel = coordVoxel
        self.diameter = diameter
        self.label = label 

srsid_path_dict = {}
for fold in range(10):
    mhdpath = "LUNA16/raw_files/subset{:d}".format(fold)
    fnamelist = []
    for fname in os.listdir(mhdpath):
        if fname.endswith(".mhd"):
            srsid_path_dict[fname[:-4]] = os.path.join(mhdpath, fname)

candidates_csv = "LUNA16/candidates_V2.csv"
cand_df = pd.read_csv(candidates_csv)

srsids = cand_df["seriesuid"].values
labels = cand_df["class"].values # 1: nodules or 0: non-nodules
coords = cand_df[["coordX", "coordY", "coordZ"]].values

max_nodules = 10
nodules_dict = {}

print("Max number of nodules for each scan: {:d}".format(max_nodules))
print("Prepare candidates for each scan.")
for i in tqdm(range(len(srsids))):
    srsid = srsids[i]
    label = labels[i]
    coord = coords[i] # coordWorld

    if not srsid in nodules_dict:
        nodules_dict[srsid] = [(coord, label)]
    else:
        if label == 1 or len(nodules_dict[srsid]) < max_nodules:
            nodules_dict[srsid].append((coord, label))


nodules_dict2 = {}
def process_one_series(srsid):
    nodules_list = nodules_dict[srsid]
    coord_list, label_list = list(zip(*nodules_list))
    coord_voxel_list = getvoxcrd(srsid, coord_list)
    nodules_dict2[srsid] = {}
    nodules_dict2[srsid]["candidates"] = []
    for idx in range(len(label_list)):
        nodule = Nodule(noduleId=srsid + "_{:d}".format(idx),
                        coordWorld=coord_list[idx],
                        coordVoxel=coord_voxel_list[idx],
                        label=label_list[idx])
        nodules_dict2[srsid]["candidates"].append(nodule)
    num_nodules = sum(label_list)
    nodules_dict2[srsid]["num_nodules"] = num_nodules
    nodules_dict2[srsid]["has_nodule"] = 1 if num_nodules > 0 else 0

from multiprocessing import Pool
pool = Pool(10)
print("Prepare nodule dictionary.")


_ = pool.map(process_one_series, nodules_dict.keys())
pool.close()
pool.join()

np.save("prepare_lung/nodule_dict.npy", nodules_dict2)
print("Save nodule dictionary to prepare_lung/nodule_dict.npy")



