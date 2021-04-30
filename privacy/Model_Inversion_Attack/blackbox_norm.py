# @Author: Zecheng He
# @Date:   2019-09-01

import time
import math
import os
import numpy as np
import h5py

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from privacy.Model_Inversion_Attack.net_norm import *
from privacy.Model_Inversion_Attack.utils_norm import *

#####################
# Training:
# python inverse_blackbox_decoder_CIFAR.py --layer ReLU22 --iter 50 --training --decodername CIFAR10CNNDecoderReLU22
#
# Testing:
# python inverse_blackbox_decoder_CIFAR.py --testing --decodername CIFAR10CNNDecoderReLU22 --layer ReLU22
#####################


def trainDecoderDNN(DATASET = 'CIFAR10', network = 'CIFAR10CNNDecoder', NEpochs = 50, imageWidth = 32,
            imageHeight = 32, imageSize = 32*32, NChannels = 3, NClasses = 10, layer = 'ReLU22', BatchSize = 32, learningRate = 1e-3,
            NDecreaseLR = 20, eps = 1e-3, AMSGrad = True, model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth", save_decoder_dir = "checkpoints/CIFAR10/",
            decodername_name = 'CIFAR10CNNDecoderReLU22', gpu = True, validation=False):

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
        
    else:
        print("Dataset unsupported")
        exit(1)

    # c1 = np.where(Y_train == 2)
    # c2 = np.where(Y_test == 2)
    # Y_train[c1] = 1
    # Y_test[c2] = 1

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    print("len(trainset) ", len(X_train))
    print("len(testset) ", len(X_test))

    print("x_trainset.shape ", X_train.shape)
    print("x_testset.shape ", X_test.shape)



###load original network
    net = torch.load(model_dir + model_name)
    if not gpu:
        net = net.cpu()
    net.eval()
    print("Validate the model accuracy...")
    if validation:
        accTest = evalTest(X_test, Y_test, net, gpu = gpu)
     
    # add_noise_inference(net.parameters(), scale=0.02)
    
    print("Validate the noisy model accuracy...")
    if validation:
        accTest = evalTest(X_test, Y_test, net, gpu = gpu)



    # net = torch.load(model_dir + model_name)
    # net.eval()
    # print "Validate the model accuracy..."
    # if validation:
    #     accTest = evalTest(X_test, Y_test, net, gpu = gpu)

### Get dims of input/output
    batchX = X_train[0]
    batchY = Y_train[0]
    batchX = torch.from_numpy(batchX)
    batchX = batchX.view(-1,1,64,64).type(torch.FloatTensor)                   # /255
    print((batchX.shape))

    if gpu:
        batchX = batchX.cuda()
    originalModelOutput = net.getLayerOutput(batchX, net.layerDict[layer]).clone()
    print((originalModelOutput.shape))


### Construct inversion network
    decoderNetDict = {
        'CIFAR10CNNDecoder':{
            'conv12': CIFAR10CNNDecoderconv1,
            'conv22': CIFAR10CNNDecoderReLU22,
            'conv32': CIFAR10CNNDecoderReLU32
        }
    }
    decoderNetFunc = decoderNetDict[network][layer]
    decoderNet = decoderNetFunc(originalModelOutput.shape[1])
    if gpu:
        decoderNet = decoderNet.cuda()

    print(decoderNet)


### MSE
    MSELossLayer = torch.nn.MSELoss()
    if gpu:
        MSELossLayer = MSELossLayer.cuda()

### parameters
        
    NBatch = int(len(X_train) / BatchSize)
    cudnn.benchmark = True
    optimizer = optim.Adam(params = decoderNet.parameters(), lr = learningRate, eps = eps, amsgrad = AMSGrad)

### Batch train
    for epoch in range(NEpochs):
        lossTrain = 0.0
        accTrain = 0.0
         
        for i in range(NBatch):
            index = np.random.randint(0,len(X_train),BatchSize)                         ###random batch

            ### process traindata and label
            batchX = X_train[index]
            batchX = torch.from_numpy(batchX)
            batchX = batchX.view(-1,1,64,64).type(torch.FloatTensor)         #    /255
            batchY = Y_train[index]
            batchY = torch.Tensor([batchY]).long().view(BatchSize)
            if gpu:
                batchX = batchX.cuda()
                batchY = batchY.cuda()

            optimizer.zero_grad()
            
            ### get the ouput and invert it
            # originalModelOutput = net.getLayerOutput(batchX, net.layerDict[layer])#.clone()
            # print originalModelOutput
            # torch.FloatTensor.abs_(originalModelOutput)
            # print originalModelOutput
            # decoderNetOutput = decoderNet.forward(originalModelOutput)
            
            decoderNetOutput = decoderNet.forward(net.getLayerOutput(batchX, net.layerDict[layer]))            
            
            # print decoderNetOutput.shape
            # print decoderNetOutput.cpu().detach().numpy()
            
            assert batchX.cpu().detach().numpy().shape == decoderNetOutput.cpu().detach().numpy().shape
            
            ### calculate the MSE of original input and inverted one
            featureLoss = MSELossLayer(batchX, decoderNetOutput)
            # print featureLoss
            # print torch.mean(decoderNetOutput ** 2)

            # feature_all = batchX - decoderNetOutput
            # feature_mean = torch.mean(feature_all - feature_all)
            # final_mean = torch.mean(feature_all - feature_mean)
            # print final_mean
            
            totalLoss = featureLoss #+ 0.4 * torch.mean(decoderNetOutput ** 2)
            

            # totalLoss = featureLoss   #+ 1 * final_mean
            # totalLoss =  final_mean
            totalLoss.backward()
            optimizer.step()

            lossTrain += featureLoss /NBatch
        
        ### claculate the mean MSE of batches 
        # print lossTrain
        lossTrain = lossTrain.cpu().detach().numpy()


### test
        # acc = 0.0
        # YBatch = 0                
        # for i in range(len(X_test)):
        #     batchx = X_test[i]
        #     batchx = torch.from_numpy(batchx)
        #     batchx = batchx.reshape(-1,1,1024,1024).type(torch.FloatTensor)
        #     batchy = Y_test[i]
        #     batchy = torch.Tensor([batchy]).long()

        #     YBatch += 1

        #     if gpu:
        #         batchx = batchx.cuda()
        #         batchy = batchy.cuda()
            
            
        #     ### get the original input and inverted one, calculate the MSE
        #     # originalModelOutput = net.getLayerOutput(batchx, net.layerDict[layer])#.clone()
        #     # decoderNetOutput = decoderNet.forward(originalModelOutput)
        #     decoderNetOutput = decoderNet.forward(net.getLayerOutput(batchx, net.layerDict[layer]))
        #     valLoss = MSELossLayer(batchx, decoderNetOutput)   
        #     # print valLoss
            
        #     acc += valLoss     
        #     # acc += np.sum(valLoss.cpu().detach().numpy())    

        # mvalLoss = acc / YBatch



        print("Epoch ", epoch, "Train Loss: ", lossTrain) #,"Test Loss: ", mvalLoss.cpu().detach().numpy()
        if (epoch + 1) % NDecreaseLR == 0:
            learningRate = learningRate / 2.0
            setLearningRate(optimizer, learningRate)

### test if the original network changes
    if validation:
        accTestEnd = evalTest(X_test, Y_test, net, gpu = gpu)
        if accTest != accTestEnd:
            print("Something wrong. Original model has been modified!")
            exit(1)

### save decoder model
    if not os.path.exists(save_decoder_dir):
        os.makedirs(save_decoder_dir)
    torch.save(decoderNet, save_decoder_dir + decodername_name)
    print("Model saved")

    newNet = torch.load(save_decoder_dir + decodername_name)
    newNet.eval()
    print("Model restore done")


def inverse(DATASET = 'CIFAR10', imageWidth = 32, inverseClass = None, imageHeight = 32,
        imageSize = 32*32, NChannels = 3, NClasses = 10, layer = 'conv11',
        model_dir = "checkpoints/CIFAR10/", model_name = "ckpt.pth", decoder_name = "CIFAR10CNNDecoderconv11.pth",
        save_img_dir = "inverted_blackbox_decoder/CIFAR10/", gpu = True, validation=False):

    print("DATASET: ", DATASET)
    print("inverseClass: ", inverseClass)

    assert inverseClass < NClasses

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
                
    else:
        print("Dataset unsupported")
        exit(1)
        
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    
    print("len(trainset) ", len(X_train))
    print("len(testset) ", len(X_test))

    print("x_trainset.shape ", X_train.shape)
    print("x_testset.shape ", X_test.shape)

###load original network

    net = torch.load(model_dir + model_name)
    if not gpu:
        net = net.cpu()
    net.eval()
    print("Validate the model accuracy...")

    if validation:
        accTest = evalTest(X_test, Y_test, net, gpu = gpu)
        
###load inversion network

    decoderNet = torch.load(model_dir + decoder_name)
    if not gpu:
        decoderNet = decoderNet.cpu()
    decoderNet.eval()
    print(decoderNet)



### load and save test data
    batchx, batchy = getImgByClass(X_test, Y_test, C = inverseClass)
    # print batchx
    
    ### save ref_img
    originalx = deprocess(batchx)
    originalx = torch.from_numpy(originalx)
    originalx = originalx.view(1,64,64)
    originalx = np.array(originalx).astype(np.uint8)
    originalx = np.squeeze(originalx,axis=0)
    originalx = Image.fromarray(originalx)
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    originalx.save( save_img_dir + str(inverseClass) + '-ref.png')        
    

    targetImg = torch.from_numpy(batchx)
    targetImg = targetImg.view(-1,1,64,64).type(torch.FloatTensor)        
    print("targetImg.size()", targetImg.size())
    batchy = torch.Tensor([batchy]).long()



### get the output and inverted input
    if gpu:
        targetImg = targetImg.cuda()
    targetLayer = net.layerDict[layer]
    refFeature = net.getLayerOutput(targetImg, targetLayer)
    print("refFeature.size()", refFeature.size())

    xGen = decoderNet.forward(refFeature)

### claculate the MSE    

    targetImg = deprocess(targetImg)
    xGen = deprocess(xGen)
    visionx1 = targetImg.clone()
    visionx2 = xGen.clone()
    MSELossLayer = torch.nn.MSELoss()
    if gpu:
        MSELossLayer = MSELossLayer.cuda()
    print("MSE 1", MSELossLayer(visionx1, visionx2).cpu().detach().numpy())
    print("MSE 2", MSELossLayer(visionx1 / 255, visionx2 / 255).cpu().detach().numpy())


### save the final result

    torchvision.utils.save_image(visionx2/255, save_img_dir + str(inverseClass) + '-inv.png')
    ximg = xGen.reshape(1,64,64).type(torch.FloatTensor).cpu().detach()
    ximg = np.array(ximg).astype(np.uint8)
    ximg = np.squeeze(ximg,axis=0)
    ximg = Image.fromarray(ximg)

    # print ximg
    ximg.show()        
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    ximg.save( save_img_dir + str(inverseClass) + '-invv.png') 
    print("Done")



### parameters
if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default = 'Medical')
        parser.add_argument('--network', type = str, default = 'CIFAR10CNNDecoder')

        parser.add_argument('--training', dest='training', action='store_true')
        parser.add_argument('--testing', dest='training', action='store_false')
        parser.set_defaults(training=False)

        parser.add_argument('--iters', type = int, default = 100)
        parser.add_argument('--eps', type = float, default = 1e-3)
        parser.add_argument('--AMSGrad', type = bool, default = True)
        parser.add_argument('--batch_size', type = int, default =1)
        parser.add_argument('--learning_rate', type = float, default = 1e-3)
        parser.add_argument('--decrease_LR', type = int, default = 10)
        parser.add_argument('--layer', type = str, default = 'conv32')
        parser.add_argument('--save_iter', type = int, default = 10)
        parser.add_argument('--inverseClass', type = int, default = 0)
        parser.add_argument('--decodername', type = str, default = "CNN_original_non_32")

        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)

        parser.add_argument('--novalidation', dest='validation', action='store_false')
        parser.set_defaults(validation=True)
        args = parser.parse_args()

        model_dir = "models/" + args.dataset + '/'
        model_name = "CNN_original.pth"
        decoder_name = args.decodername + '.pth'

        save_img_dir = "CNN_original" + args.dataset + '/' + args.layer + '/'

        if args.dataset == 'Medical':

            imageWidth = 64
            imageHeight =64
            imageSize = imageWidth * imageHeight
            NChannels = 1
            NClasses = 2

        else:
            print("No Dataset Found")
            exit()

### train
        trainDecoderDNN(DATASET = args.dataset, network = 'CIFAR10CNNDecoder', NEpochs = args.iters, imageWidth = imageWidth,
        imageHeight = imageHeight, imageSize = imageSize, NChannels = NChannels, NClasses = NClasses, layer = args.layer, BatchSize = args.batch_size, learningRate = args.learning_rate,
        NDecreaseLR = args.decrease_LR, eps = args.eps, AMSGrad = True, model_dir = "models/Medical/", model_name = model_name, save_decoder_dir = "models/Medical/",
        decodername_name = decoder_name, gpu = args.gpu, validation=args.validation)

### test
        for c in range(NClasses):
            inverse(DATASET = args.dataset, imageHeight = imageHeight, imageWidth = imageWidth, inverseClass = c,
            imageSize = imageSize, NChannels = NChannels, NClasses = NClasses, layer = args.layer,
            model_dir = model_dir, model_name = model_name, decoder_name = decoder_name,
            save_img_dir = save_img_dir, gpu = args.gpu, validation=args.validation)


    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
