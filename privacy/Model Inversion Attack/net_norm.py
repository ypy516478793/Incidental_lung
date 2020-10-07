# @Author: zechenghe
# @Date:   2019-01-21T12:17:31-05:00
# @Last modified by:   zechenghe
# @Last modified time: 2019-02-01T14:01:15-05:00

import time
import math
import os
import sys
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import collections




## 64*64

class CIFAR10CNN(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNN, self).__init__()
        self.features = []
        self.layerDict = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 32,
            kernel_size = 3,
            padding = 1,
            stride = 1
        )
        self.features.append(self.conv11)
        self.layerDict['conv11'] = self.conv11

        self.ReLU11 = nn.ReLU()
        self.features.append(self.ReLU11)
        self.layerDict['ReLU11'] = self.ReLU11

        self.conv12 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = 3,
            padding = 1,
            stride = 1
        )
        self.features.append(self.conv12)
        self.layerDict['conv12'] = self.conv12

        self.ReLU12 = nn.ReLU()
        self.features.append(self.ReLU12)
        self.layerDict['ReLU12'] = self.ReLU12

        self.pool1 = nn.MaxPool2d(2,2)
        self.features.append(self.pool1)
        self.layerDict['pool1'] = self.pool1


        self.conv21 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv21)
        self.layerDict['conv21'] = self.conv21

        self.ReLU21 = nn.ReLU()
        self.features.append(self.ReLU21)
        self.layerDict['ReLU21'] = self.ReLU21




        self.conv22 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
            # stride =2
        )
        self.features.append(self.conv22)
        self.layerDict['conv22'] = self.conv22

        self.ReLU22 = nn.ReLU()
        self.features.append(self.ReLU22)
        self.layerDict['ReLU22'] = self.ReLU22

        self.pool2 = nn.MaxPool2d(2,2)
        self.features.append(self.pool2)
        self.layerDict['pool2'] = self.pool2

        self.conv31 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv31)
        self.layerDict['conv31'] = self.conv31

        self.ReLU31 = nn.ReLU()
        self.features.append(self.ReLU31)
        self.layerDict['ReLU31'] = self.ReLU31

        self.conv32 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv32)
        self.layerDict['conv32'] = self.conv32


        self.ReLU32 = nn.ReLU()
        self.features.append(self.ReLU32)
        self.layerDict['ReLU32'] = self.ReLU32

        self.pool3 = nn.MaxPool2d(2,2)
        self.features.append(self.pool3)
        self.layerDict['pool3'] = self.pool3

        self.classifier = []

        self.feature_dims = 32 * 8 * 8
        self.fc1 = nn.Linear(self.feature_dims, 256)
        self.classifier.append(self.fc1)
        self.layerDict['fc1'] = self.fc1

        self.fc1act = nn.Sigmoid()
        self.classifier.append(self.fc1act)
        self.layerDict['fc1act'] = self.fc1act

        self.fc2 = nn.Linear(256, 2)
        self.classifier.append(self.fc2)
        self.layerDict['fc2'] = self.fc2

    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        x = x.view(-1, self.feature_dims)

        for layer in self.classifier:
            x = layer(x)
        # x = F.log_softmax(x, dim=1)
        return x


    def forward_from(self, x, layer):

        if layer in self.layerDict:
            targetLayer = self.layerDict[layer]

            if targetLayer in self.features:
                layeridx = self.features.index(targetLayer)
                for func in self.features[layeridx+1:]:
                    x = func(x)

                x = x.view(-1, self.feature_dims)
                for func in self.classifier:
                    x = func(x)
                return x

            else:
                layeridx = self.classifier.index(targetLayer)
                for func in self.classifier[layeridx:]:
                    x = func(x)
                return x
        else:
            print "layer not exists"
            exit(1)


    def getLayerOutput(self, x, targetLayer):
        for layer in self.features:
            x = layer(x)
            if layer == targetLayer:
                return x

        x = x.view(-1, self.feature_dims)
        for layer in self.classifier:
            x = layer(x)
            if layer == targetLayer:
                return x

        # Should not go here
        print "Target layer not found"
        exit(1)








### 256


class CIFAR10CNNDecoderconv1(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNNDecoderconv1, self).__init__()
        self.decoder = []
        self.layerDict = collections.OrderedDict()
        self.deconv1 = nn.ConvTranspose2d(
            in_channels = 32,
            out_channels = 16,
            kernel_size = 3,
            padding = 1
        )

        self.layerDict['deconv1'] = self.deconv1
        self.ReLU1 = nn.ReLU()
        self.layerDict['ReLU1'] = self.ReLU1
        self.deconv2 = nn.ConvTranspose2d(
            in_channels = 16,
            out_channels = 1,
            kernel_size = 3,
            stride = 1,
            padding =1
            # output_padding = 1
        )
        self.layerDict['deconv2'] = self.deconv2
    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x




class CIFAR10CNNDecoderReLU22(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNNDecoderReLU22, self).__init__()

        self.layerDict = collections.OrderedDict()

        self.deconv11 = nn.ConvTranspose2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding = 1
        )
        self.layerDict['deconv11'] = self.deconv11

        self.ReLU11 = nn.ReLU()
        self.layerDict['ReLU11'] = self.ReLU11

        self.deconv21 = nn.ConvTranspose2d(
            in_channels = 32,
            out_channels = 1,
            kernel_size = 3,
            stride = 1,
            padding = 1
            # output_padding =1

        )

        self.layerDict['deconv21'] = self.deconv21

    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x


class CIFAR10CNNDecoderReLU32(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNNDecoderReLU32, self).__init__()

        self.layerDict = collections.OrderedDict()

        self.deconv11 = nn.ConvTranspose2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding = 1
        )
        self.layerDict['deconv11'] = self.deconv11

        self.ReLU11 = nn.ReLU()
        self.layerDict['ReLU11'] = self.ReLU11

        self.deconv21 = nn.ConvTranspose2d(
            in_channels = 32,
            out_channels = 16,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            output_padding = 1
        )
        self.layerDict['deconv21'] = self.deconv21

        self.ReLU21 = nn.ReLU()
        self.layerDict['ReLU21'] = self.ReLU21

        self.deconv31 = nn.ConvTranspose2d(
            in_channels = 16,
            out_channels = 1,
            kernel_size = 3,
            stride =1,
            padding = 1,
            # output_padding = 1
        )

        self.layerDict['deconv31'] = self.deconv31

    def forward(self, x):
        for layer in self.layerDict:
            x = self.layerDict[layer](x)
        return x






### 1024


# class CIFAR10CNNDecoderconv1(nn.Module):
#     def __init__(self, NChannels):
#         super(CIFAR10CNNDecoderconv1, self).__init__()
#         self.decoder = []
#         self.layerDict = collections.OrderedDict()
#         self.deconv1 = nn.ConvTranspose2d(
#             in_channels = 32,
#             out_channels = 16,
#             kernel_size = 3,
#             padding = 1
#         )

#         self.layerDict['deconv1'] = self.deconv1
#         self.ReLU1 = nn.ReLU()
#         self.layerDict['ReLU1'] = self.ReLU1
#         self.deconv2 = nn.ConvTranspose2d(
#             in_channels = 16,
#             out_channels = 1,
#             kernel_size = 3,
#             stride = 1,
#             padding =1
#             # output_padding = 1
#         )
#         self.layerDict['deconv2'] = self.deconv2
#     def forward(self, x):
#         for layer in self.layerDict:
#             x = self.layerDict[layer](x)
#         return x






# class CIFAR10CNNDecoderReLU22(nn.Module):
#     def __init__(self, NChannels):
#         super(CIFAR10CNNDecoderReLU22, self).__init__()

#         self.layerDict = collections.OrderedDict()

#         self.deconv11 = nn.ConvTranspose2d(
#             in_channels = 32,
#             out_channels = 32,
#             kernel_size = 3,
#             stride = 2,
#             padding = 1,
#             output_padding = 1
#         )
#         self.layerDict['deconv11'] = self.deconv11

#         self.ReLU11 = nn.ReLU()
#         self.layerDict['ReLU11'] = self.ReLU11

#         self.deconv21 = nn.ConvTranspose2d(
#             in_channels = 32,
#             out_channels = 1,
#             kernel_size = 3,
#             stride = 2,
#             padding = 1,
#             output_padding =1

#         )

#         self.layerDict['deconv21'] = self.deconv21

#     def forward(self, x):
#         for layer in self.layerDict:
#             x = self.layerDict[layer](x)
#         return x


# class CIFAR10CNNDecoderReLU32(nn.Module):
#     def __init__(self, NChannels):
#         super(CIFAR10CNNDecoderReLU32, self).__init__()

#         self.layerDict = collections.OrderedDict()

#         self.deconv11 = nn.ConvTranspose2d(
#             in_channels = 32,
#             out_channels = 32,
#             kernel_size = 5,
#             stride = 4,
#             padding = 1,
#             output_padding = 1
#         )
#         self.layerDict['deconv11'] = self.deconv11

#         self.ReLU11 = nn.ReLU()
#         self.layerDict['ReLU11'] = self.ReLU11

#         self.deconv21 = nn.ConvTranspose2d(
#             in_channels = 32,
#             out_channels = 16,
#             kernel_size = 3,
#             stride = 2,
#             padding = 1,
#             output_padding = 1
#         )
#         self.layerDict['deconv21'] = self.deconv21

#         self.ReLU21 = nn.ReLU()
#         self.layerDict['ReLU21'] = self.ReLU21

#         self.deconv31 = nn.ConvTranspose2d(
#             in_channels = 16,
#             out_channels = 1,
#             kernel_size = 3,
#             stride =2,
#             padding = 1,
#             output_padding = 1
#         )

#         self.layerDict['deconv31'] = self.deconv31

#     def forward(self, x):
#         for layer in self.layerDict:
#             x = self.layerDict[layer](x)
#         return x
