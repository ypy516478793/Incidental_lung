import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DistillLayer(nn.Module):
    def __init__(
        self,
        emb_func,
        classifier,
        is_distill,
        emb_func_path=None,
        classifier_path=None,
    ):
        super(DistillLayer, self).__init__()
        self.emb_func = self._load_state_dict(emb_func, emb_func_path, is_distill)
        self.classifier = self._load_state_dict(classifier, classifier_path, is_distill)

    def _load_state_dict(self, model, state_dict_path, is_distill):
        new_model = None
        if is_distill and state_dict_path is not None:
            new_model = copy.deepcopy(model)
            model_state_dict = torch.load(state_dict_path, map_location="cpu")
            new_model.load_state_dict(model_state_dict)
        return new_model

    @torch.no_grad()
    def forward(self, x):
        output = None
        if self.emb_func is not None and self.classifier is not None:
            output = self.emb_func(x)
            output = self.classifier(output)
        return output


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400,
                 clinical=False,):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clinical = clinical


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.clinical:
            x, x_clinical = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)


        return x

class DistillKLLoss(nn.Module):
    def __init__(self, T):
        super(DistillKLLoss, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        if y_t is None:
            return 0.0

        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.size(0)
        return loss

def topk_(matrix, K, axis):
    """
    the function to calc topk acc of ndarrary.

    TODO

    """
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
    return topk_data_sort, topk_index_sort

def accuracy(output, target, topk=1):
    """
    Calc the acc of tpok.

    output and target have the same dtype and the same shape.

    Args:
        output (torch.Tensor or np.ndarray): The output.
        target (torch.Tensor or np.ndarray): The target.
        topk (int or list or tuple): topk . Defaults to 1.

    Returns:
        float: acc.
    """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = {
            "Tensor": torch.topk,
            "ndarray": lambda output, maxk, axis: (
                None,
                torch.from_numpy(topk_(output, maxk, axis)[1]).to(target.device),
            ),
        }[output.__class__.__name__](output, topk, 1)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
        res = correct_k.mul_(100.0 / batch_size).item()
        return res

from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv2d") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv2d") != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find("Linear") != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv2d") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv2d") != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type="kaiming"):
    # print("initialization method [%s]" % init_type)
    if init_type == "normal":
        net.apply(weights_init_normal)
    elif init_type == "xavier":
        net.apply(weights_init_xavier)
    elif init_type == "kaiming":
        net.apply(weights_init_kaiming)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError

class ResNet_RFS(nn.Module):
    def __init__(
        self,
        feat_dim,
        num_class,
        gamma=1,
        alpha=0,
        is_distill=False,
        kd_T=4,
        emb_func_path=None,
        classifier_path=None,
        clinical=False,
        init_type="kaiming",
        device="cpu"
    ):
        super(ResNet_RFS, self).__init__()

        self.meta = True

        self.feat_dim = feat_dim
        self.num_class = num_class

        self.is_distill = is_distill
        self.gamma = gamma
        self.alpha = alpha
        self.init_type = init_type

        self.way_num = num_class
        self.shot_num = 1
        self.query_num = 7
        self.device = device

        # self.classifier = nn.Linear(self.feat_dim, self.num_class)
        self.emb_func = generate_model(18, n_input_channels=1, n_classes=num_class)
        if clinical:
            self.fc_clinical = nn.Sequential(
                nn.Linear(37, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(feat_dim + 256, num_class)
        else:
            self.classifier = nn.Linear(feat_dim, num_class)

        self.ce_loss_func = nn.CrossEntropyLoss()
        self.kl_loss_func = DistillKLLoss(T=kd_T)

        self._init_network()

        self.distill_layer = DistillLayer(
            self.emb_func,
            self.classifier,
            self.is_distill,
            emb_func_path,
            classifier_path,
        )

    def _init_network(self):
        init_weights(self, self.init_type)

    def _generate_local_targets(self, episode_size):
        local_targets = (
            torch.arange(self.way_num, dtype=torch.long)
            .view(1, -1, 1)
            .repeat(episode_size, 1, self.shot_num + self.query_num)
            .view(-1)
        )
        return local_targets


    def forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        # image = image.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(image)
        # support_feat, query_feat, support_target, query_target = self.split_by_episode(feat, global_target)

        # support_feat, query_feat, support_target, query_target = train_test_split(feat, global_target, test_size=0.2)
        support_ids, query_ids = train_test_split(np.arange(len(feat)))
        support_feat = feat[support_ids].unsqueeze(0)
        query_feat = feat[query_ids].unsqueeze(0)
        support_target = global_target[support_ids].unsqueeze(0)
        query_target = global_target[query_ids].unsqueeze(0)

        episode_size = support_feat.size(0)

        output_list = []
        acc_list = []
        prob_list = []
        qt_list = []
        for idx in range(episode_size):
            SF = support_feat[idx]
            QF = query_feat[idx]
            ST = support_target[idx]
            QT = query_target[idx]

            classifier = self.set_forward_adaptation(SF, ST)

            QF = F.normalize(QF, p=2, dim=1).detach().cpu().numpy()
            QT = QT.detach().cpu().numpy()

            output = classifier.predict(QF)
            prob = classifier.predict_proba(QF)
            acc = metrics.accuracy_score(QT, output) * 100

            output_list.append(output)
            acc_list.append(acc)
            qt_list.append(QT)
            prob_list.append(prob)

        output = np.stack(output_list, axis=0)
        acc = sum(acc_list) / episode_size
        qt_list = np.concatenate(qt_list, axis=0)
        prob_list = np.concatenate(prob_list, axis=0)

        prob_list = torch.from_numpy(prob_list).to(self.device)
        qt_list = torch.from_numpy(qt_list).to(self.device)

        # return output, acc, qt_list, prob_list
        return prob_list, qt_list

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        # image = image.to(self.device)
        # global_target = global_target.to(self.device)

        feat = self.emb_func(image)

        if self.clinical:
            x, x_clinical = batch
            x_clinical = self.fc_clinical(x_clinical)
            x = torch.cat([x, x_clinical], 1)
            output = self.classifier(x)
        else:
            output = self.classifier(batch)

        # output = self.classifier(feat)
        distill_output = self.distill_layer(image)

        gamma_loss = self.ce_loss_func(output, global_target)
        alpha_loss = self.kl_loss_func(output, distill_output)
        loss = gamma_loss * self.gamma + alpha_loss * self.alpha

        acc = accuracy(output, global_target)

        return output, acc, loss

    def set_forward_adaptation(self, support_feat, support_target):
        classifier = LogisticRegression(
            penalty="l2",
            random_state=0,
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            multi_class="multinomial",
        )

        support_feat = F.normalize(support_feat, p=2, dim=1).detach().cpu().numpy()
        support_target = support_target.detach().cpu().numpy()

        classifier.fit(support_feat, support_target)

        return classifier

    def train(self, mode=True):
        self.emb_func.train(mode)
        self.classifier.train(mode)
        self.distill_layer.train(False)

    def eval(self):
        super(ResNet_RFS, self).eval()


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = generate_model(18, n_input_channels=1, n_classes=2).to(device)
    summary(model, (1, 64, 64, 64))
    print()