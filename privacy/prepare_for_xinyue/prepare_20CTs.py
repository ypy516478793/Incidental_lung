
from dataLoader.IncidentalData import LungDataset
from utils.model_utils import plot_bbox, center_stack

import numpy as np
import os




rootFolder = "../data/"
pos_label_file = "../data/pos_labels.csv"
cat_label_file = "../data/Lung Nodule Clinical Data_Min Kim (No name).xlsx"
cube_size = 64
trainData = LungDataset(rootFolder, labeled_only=True, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
                        cube_size=cube_size, reload=False, train=True)

posIds, negIds = [], []
num_samples_per_class = 10
for i in trainData.imageIds:
    cat = trainData.load_cat(i)
    if cat == 0:
        posIds.append(i)
    else:
        negIds.append(i)

posIds = np.random.choice(posIds, num_samples_per_class+5, replace=False)
negIds = np.random.choice(negIds, num_samples_per_class, replace=False)
posSamplesInfos = trainData.imageInfo[posIds]
negSamplesInfos = trainData.imageInfo[negIds]

# for i in posIds:
#     image = trainData.load_image(i)
#     cat = trainData.load_cat(i)
#     assert cat == 0
#     pos = trainData.load_pos(i)
#     assert pos.shape == (1, 4)
#
#     savedir = "posCases/"
#     os.makedirs(os.path.dirname(savedir), exist_ok=True)
#     savePath = os.path.join(savedir, "sample{:d}.npz".format(i))
#
#     np.savez_compressed(savePath, image=image, pos=pos[0])
#     print("Save scan cube to {:s}".format(savePath))
#
#     crop_size = 64
#     center = crop_size // 2
#     cubes = trainData.get_cube(i, crop_size)
#
#     savedir = "Figures/posCases/"
#     os.makedirs(os.path.dirname(savedir), exist_ok=True)
#     savePath = os.path.join(savedir, "sample{:d}".format(i))
#
#     for cube, p in zip(cubes, pos):
#         plot_bbox(cube, np.array([center, center, center, p[-1]]), savePath, show=False)
#         center_stack(cube, p[-1], savePath, show=False)

for i in negIds:
    image = trainData.load_image(i)
    cat = trainData.load_cat(i)
    assert cat == 1
    pos = trainData.load_pos(i)
    assert pos.shape == (1, 4)

    savedir = "negCases/"
    os.makedirs(os.path.dirname(savedir), exist_ok=True)
    savePath = os.path.join(savedir, "sample{:d}.npz".format(i))

    np.savez_compressed(savePath, image=image, pos=pos[0])
    print("Save scan cube to {:s}".format(savePath))

    crop_size = 64
    center = crop_size // 2
    cubes = trainData.get_cube(i, crop_size)

    savedir = "Figures/negCases/"
    os.makedirs(os.path.dirname(savedir), exist_ok=True)
    savePath = os.path.join(savedir, "sample{:d}".format(i))

    for cube, p in zip(cubes, pos):
        plot_bbox(cube, np.array([center, center, center, p[-1]]), savePath, show=False)
        center_stack(cube, p[-1], savePath, show=False)