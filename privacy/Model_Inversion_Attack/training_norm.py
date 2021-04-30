import time
import math
import os
import numpy as np
import h5py
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from privacy.Model_Inversion_Attack.net_norm import *
from privacy.Model_Inversion_Attack.utils_norm import *
# from preprocess_test import *

def train(DATASET = 'CIFAR10', network = 'MIASCNN', NEpochs = 200, imageWidth = 32,
        imageHeight = 32, imageSize = 32*32, NChannels = 3, NClasses = 10,
        BatchSize = 32, learningRate = 1e-3, NDecreaseLR = 20, eps = 1e-3,
        AMSGrad = True, model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth", gpu = True):

    print("DATASET: ", DATASET)


###load data

    if DATASET == 'Medical':  
        
        h5f = h5py.File('/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/privacy/Model_Inversion_Attack/lung_train76.h5', 'r')
        X_train = h5f['x'][:]
        Y_train = h5f['y'][:]
        h5f.close()

        h5f = h5py.File('/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/privacy/Model_Inversion_Attack/lung_test26.h5', 'r')
        X_test = h5f['x'][:]
        Y_test = h5f['y'][:]
        h5f.close()
        

        print(Y_train)
        print(Y_test)
        X_train = preprocess(X_train)
        X_test = preprocess(X_test)

###load network

    netDict = {
        'CIFAR10CNN': CIFAR10CNN,
    }

    if network in netDict:               
        net = netDict[network](NChannels)
    else:
        print("Network not found")
        exit(1)

    print(net)
    print("len(trainset) ", len(X_train))
    print("len(testset) ", len(X_test))

### pararmeters
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    if gpu:                                               # GPU
        net.cuda()
        criterion.cuda()
        softmax.cuda()

    optimizer = optim.Adam(params = net.parameters(), lr = learningRate, eps = eps, amsgrad = AMSGrad) # Adam optimizer
    # optimizer = torch.optim.SGD(net.parameters(), lr=learningRate)                        # SGD optimizer
    NBatch = int(len(X_train) / BatchSize)
    cudnn.benchmark = True

###training
      
    for epoch in range(NEpochs):
        lossTrain = 0.0
        accTrain = 0.0
        
        # if epoch >100:
        #      optimizer = optim.Adam(params = net.parameters(), lr = learningRate/2, eps = eps, amsgrad = AMSGrad)
        
        for i in range(NBatch):
            
            ### batch data (shuffle)
            index = np.random.randint(0, len(X_train), BatchSize)
            batchX = X_train[index]
            batchX = torch.from_numpy(batchX)
            batchX = batchX.view(-1,1,64,64).type(torch.FloatTensor) 
            
            batchY = Y_train[index]
           
            batchY = torch.Tensor([batchY]).long().view(BatchSize)
            
            
            if gpu:
                batchX = batchX.cuda()
                batchY = batchY.cuda()
 
            ### calculate the loss and gradients                                     
            optimizer.zero_grad()
            logits = net.forward(batchX)
            prob = softmax(logits)

            # print prob
            loss = criterion(logits, batchY)    # + l2loss(logits)   # softmax            
            # loss = F.nll_loss(F.log_softmax(logits, dim=1), batchY) #+ 0.001 * l2loss(logits)   # logsoftmax
            loss.backward()

            ### add noise            
            # clip_grad(net.parameters(), grad_norm_bound=5)
            # add_noise(net.parameters(), grad_norm_bound=5, scale=10)
            
            optimizer.step()
          
            lossTrain += loss.cpu().detach().numpy() / NBatch

            if gpu:
                pred = np.argmax(prob.cpu().detach().numpy(), axis = 1)
                groundTruth = batchY.cpu().detach().numpy()

            else:
                pred = np.argmax(prob.detach().numpy(), axis = 1)
                groundTruth = batchY.detach().numpy()
                
                
          
            acc = np.mean(pred == groundTruth)   # accuracy per Batch
            accTrain += acc / NBatch            # accuracy per iteration
            
        


        if (epoch + 1) % NDecreaseLR == 0:
            learningRate = learningRate / 2.0
            setLearningRate(optimizer, learningRate)

        print("Epoch: ", epoch, "Loss: ", lossTrain, "Train accuracy: ", accTrain)

### Testing

        accTest = evalTest(X_test, Y_test, net, gpu = gpu)
        
### save trained network

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(net, model_dir + model_name)
    print("Model saved")

### load trained network

    newNet = torch.load(model_dir + model_name)
    newNet.eval()
    accTest = evalTest(X_test, Y_test, net, gpu = gpu)
    print("Model restore done")


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'Medical')
        parser.add_argument('--network', type = str, default = 'CIFAR10CNN')
        parser.add_argument('--epochs', type = int, default = 50)
        parser.add_argument('--eps', type = float, default = 1e-8)   #-3
        parser.add_argument('--AMSGrad', type = bool, default = True)
        parser.add_argument('--batch_size', type = int, default = 1)
        parser.add_argument('--learning_rate', type = float, default = 1e-3)
        parser.add_argument('--decrease_LR', type = int, default = 20)   #20

        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)
        args = parser.parse_args()

        model_dir = "models/" + args.dataset + '/'
        model_name = "CNN_original.pth"

        if args.dataset == 'Medical':

            imageWidth = 64
            imageHeight = 64
            imageSize = imageWidth * imageHeight
            NChannels = 1
            NClasses = 2
            network = 'CIFAR10CNN'


        else:
            print("No Dataset Found")
            exit(0)

        
        train(DATASET = args.dataset, network = network, NEpochs = args.epochs, imageWidth = imageWidth,
        imageHeight = imageHeight, imageSize = imageSize, NChannels = NChannels, NClasses = NClasses,
        BatchSize = args.batch_size, learningRate = args.learning_rate, NDecreaseLR = args.decrease_LR, eps = args.eps,
        AMSGrad = args.AMSGrad, model_dir = model_dir, model_name = model_name, gpu = args.gpu)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
