from dataLoader.IncidentalData import LungDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from classifier.mlp import Net
import math
import itertools


def infer(testLoader, model, device, model_folder, values, freq):

    epoch = 30
    model.load_state_dict(torch.load(os.path.join(model_folder, 'epoch_' + str(epoch) + '.pt'), map_location=torch.device('cpu')))
    print("load model from: {:s}".format(os.path.join(model_folder, 'epoch_' + str(epoch) + '.pt')))
    model.eval()

    num_classes = 2
    scores = []
    correct = 0
    total = 0
    all_pred = []
    all_label = []
    target_cols = [0, 11, 13, 14, 17]
    guesses = np.zeros((len(target_cols)))
    for sample_batch in testLoader:
        x2, y2 = sample_batch["features"].float(), sample_batch["label"]

        x_np = x2.detach().numpy()
        infer_x = attribute_invert(model, x_np, y2, target_cols, values, freq, device)
        for i in range(len(target_cols)):
            comp = infer_x == x_np
            if comp[0, target_cols[i]] == True:
                guesses[i] += 1
    return guesses


def attribute_invert(model, X, y, target_cols, values, freq, device):
    guesses = []
    num_variants = len(target_cols) # need to be binary
    row_X = np.vstack([X] * int(math.pow(2,num_variants)))  # create copies of X
    lst = [list(i) for i in itertools.product([0, 1], repeat=num_variants)]

    total_prob = np.zeros(len(lst))
    for i in range(len(lst)):
        prob = np.zeros(len(target_cols))
        for j in range(len(target_cols)):
            if lst[i][j] == 0:
                lst[i][j] = values[j][0]
                prob[j] = freq[j][0]
            else:
                lst[i][j] = values[j][1]
                prob[j] = freq[j][1]
        row_prob = np.prod(prob)
        total_prob[i] = row_prob
    lst = np.array(lst)

    for index, col in enumerate(target_cols):
        for jj in range(len(row_X)):
            row_X[jj,col] = lst[jj,index]

    row_X_tensor = torch.from_numpy(row_X)
    p = model(row_X_tensor.to(device)).cpu().detach().numpy()
    p_modify = np.multiply(p, np.stack((total_prob,total_prob),axis=-1))
    guess_row = np.where(p_modify == np.max(p_modify))[0][0]
    return row_X[guess_row,:]


def extract_freq(dataLoader, target_cols):
    values = np.zeros((len(target_cols),2))
    counts = np.zeros((len(target_cols),2))
    for allData in dataLoader:
        x, y = allData["features"].float(), allData["label"]
        x = x.detach().numpy()
        for i, col in enumerate(target_cols):
            target = x[:,col]
            value, count = np.unique(target, return_counts=True)
            values[i] = value
            counts[i] = count

    freq = counts/len(x)
    return values, freq


def main():

    # use_clinical_features = True
    # rootFolder = "prepare_for_xinyue/"
    # cat_label_file = "prepare_for_xinyue/clinical_info.xlsx"
    # load_model_folder = "../classifier/model/102patients_new/Multilayer perceptron_clinical/"
    # cube_size = 64
    # allData = LungDataset(rootFolder, cat_label_file=cat_label_file, cube_size=cube_size, clinical=use_clinical_features)

    rootFolder = "../data/"
    pos_label_file = "../data/pos_labels.csv"
    cat_label_file = "../data/Lung Nodule Clinical Data_Min Kim (No name).xlsx"
    load_model_folder = "../classifier/model/102patients_new/Multilayer perceptron_clinical/"
    cube_size = 64
    use_clinical_features = True
    allData = LungDataset(rootFolder, labeled_only=True, pos_label_file=pos_label_file, cat_label_file=cat_label_file,
                           cube_size=cube_size, reload=False, train=None, screen=True, clinical=use_clinical_features)

    allLoader = DataLoader(allData, batch_size=len(allData), shuffle=True)
    target_cols = [0, 11, 13, 14, 17]
    target_attr_names = allData.cat_df.columns[[x + 1 for x in target_cols]].to_numpy()
    values, freq = extract_freq(allLoader, target_cols)
    attackLoader = DataLoader(allData, batch_size=1, shuffle=True)

    modelName = "Multilayer_perceptron"
    model = Net(26, output_dim=2)
    print("Use model: {:s}".format(modelName))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    attack_count = infer(attackLoader, model, device, load_model_folder, values, freq)
    attack_acc = attack_count/len(allData)
    print("Attack Attributes: ", target_attr_names)
    print("Attack Accuracy: ", attack_acc)
    plt.figure()
    plt.bar(target_attr_names, attack_acc)
    plt.xticks(rotation=-8)
    plt.ylim([0,1])
    plt.show()


if __name__ == '__main__':
    main()