import os
import numpy as np
import argparse
import torch

torch.multiprocessing.freeze_support()
torch.set_num_threads(1)
import torch.optim as optim
from tqdm import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
# import matplotlib.pyplot as plt
from option import Options
from datasets import oneShotBaseCls_84
from datasets import miniimagenet as mini
from pathlib import Path
import logging
import datetime as dt
import sys

args = Options().parse()
from torch.optim import lr_scheduler
import copy
import time


## logging
def get_logger(name, log_dir="log/"):
    log_path = Path(log_dir)
    path = log_path
    path.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger("Main")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        path / (dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + name + ".log"))
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Clip(object):
    def __init__(self):
        return

    def __call__(self, tensor):
        t = tensor.clone()
        t[t > 1] = 1
        t[t < 0] = 0
        return t


def plotPicture(image, name, detransform):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    A = image.clone()
    ax.imshow(detransform(A))
    fig.savefig('picture/' + str(name) + '.png')
    plt.close(fig)


######################################################################
# Define the Embedding Network
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# class ClassificationNetwork(nn.Module):
#     def __init__(self):
#         super(ClassificationNetwork, self).__init__()
#         self.convnet = torchvision.models.resnet18(pretrained=False)
#         num_ftrs = self.convnet.fc.in_features
#         self.convnet.fc = nn.Linear(num_ftrs,64)
#         #print(self.convnet)

#     def forward(self,inputs):
#         outputs = self.convnet(inputs)

#         return outputs

class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()
        self.convnet = Conv6()
        num_ftrs = self.convnet.final_feat_dim
        self.convnet.fc = nn.Linear(num_ftrs, 64)

    def forward(self, inputs):
        outputs = self.convnet(inputs)

        return outputs


# resnet18 without fc layer
import math


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = BatchNorm2d_fw(outdim)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, flatten=True):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self, x):
        out = self.trunk(x)
        return out


def Conv4():
    return ConvNet(4)


def Conv6():
    return ConvNet(6)


def weightNet():
    return ConvNet(6)


##############################


# def conv_block(in_channels, out_channels):
#     bn = nn.BatchNorm2d(out_channels)
#     nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=1),
#         bn,
#         nn.ReLU(),
#         nn.MaxPool2d(2)
#     )


# class weightNet(nn.Module):
#     def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
#         super(weightNet, self).__init__()
#         self.encoder = nn.Sequential(
#             conv_block(x_dim, hid_dim),
#             conv_block(hid_dim, hid_dim),
#             conv_block(hid_dim, hid_dim),
#             conv_block(hid_dim, z_dim),
#         )

#         self.out_channels = 1600

#     def forward(self,x):

#         x = self.encoder(x)
#         return x.view(x.size(0), -1)


# class weightNet(nn.Module):
#     def __init__(self):
#         super(weightNet, self).__init__()
#         self.resnet = ClassificationNetwork()
#         # self.resnet.load_state_dict(torch.load('models/'+str(args.network)+'.t7', map_location=lambda storage, loc: storage))
#         # print('loading ',str(args.network))

#         self.conv1 = self.resnet.convnet.conv1
#         self.conv1.load_state_dict(self.resnet.convnet.conv1.state_dict())
#         self.bn1 = self.resnet.convnet.bn1
#         self.bn1.load_state_dict(self.resnet.convnet.bn1.state_dict())
#         self.relu = self.resnet.convnet.relu
#         self.maxpool = self.resnet.convnet.maxpool
#         self.layer1 = self.resnet.convnet.layer1
#         self.layer1.load_state_dict(self.resnet.convnet.layer1.state_dict())
#         self.layer2 = self.resnet.convnet.layer2
#         self.layer2.load_state_dict(self.resnet.convnet.layer2.state_dict())
#         self.layer3 = self.resnet.convnet.layer3
#         self.layer3.load_state_dict(self.resnet.convnet.layer3.state_dict())
#         self.layer4 = self.resnet.convnet.layer4
#         self.layer4.load_state_dict(self.resnet.convnet.layer4.state_dict())
#         self.layer4 = self.resnet.convnet.layer4
#         self.layer4.load_state_dict(self.resnet.convnet.layer4.state_dict())
#         self.avgpool = self.resnet.convnet.avgpool

#     def forward(self,x):

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         layer1 = self.layer1(x) # (, 64L, 56L, 56L)
#         layer2 = self.layer2(layer1) # (, 128L, 28L, 28L)
#         layer3 = self.layer3(layer2) # (, 256L, 14L, 14L)
#         layer4 = self.layer4(layer3) # (,512,7,7)
#         x = self.avgpool(layer4) # (,512,1,1)
#         x = x.view(x.size(0), -1)
#         return x


# class smallNet(nn.Module):
#     def __init__(self):
#         super(smallNet, self).__init__()
#         def conv_block(in_channels, out_channels):
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2)
#             )

#         self.encoder = nn.Sequential( # 6*224*224
#             conv_block(6, 32), # 64*112*112
#             conv_block(32, 64), # 64*56*56
#             conv_block(64, 64), # 64*28*28
#             conv_block(64, 32), # 64*14*14
#             conv_block(32, 16), # 32*7*7
#             Flatten() # 784
#         )
#         print(self.encoder)

#     def forward(self,inputs):

#         """                 
#         inputs: Batchsize*3*224*224
#         outputs: Batchsize*100
#         """
#         outputs = self.encoder(inputs)

#         return outputs

class smallNet(nn.Module):
    def __init__(self, flatten=True):
        super(smallNet, self).__init__()
        trunk = []
        for i in range(6):
            indim = 6 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self, x):
        out = self.trunk(x)
        return out


class GNet(nn.Module):
    '''
        Two branch's performance are similar one branch's
        So we use one branch here
        Deeper attention network do not bring in benifits
        So we use small network here
    '''

    def __init__(self):
        super(GNet, self).__init__()
        # self.ANet = weightNet()
        # self.BNet = weightNet()
        self.attentionNet = smallNet()

        self.toWeight = nn.Sequential(
            nn.Linear(1600, args.Fang * args.Fang),
            # nn.ReLU(),
            # nn.Linear(100,args.Fang*args.Fang),
            # nn.Linear(1024,9),
            # nn.Tanh(),
            # nn.ReLU(),
        )

        self.CNet = weightNet()
        self.fc = nn.Linear(1600, 64)

        if str(args.network) != 'None':
            print('loading ', str(args.network))
            conv6 = ClassificationNetwork()
            conv6.load_state_dict(torch.load('models/' + str(args.network) + '.t7', map_location=lambda storage, loc: storage))
            self.CNet.trunk.load_state_dict(conv6.convnet.trunk.state_dict())
            self.fc.load_state_dict(conv6.convnet.fc.state_dict())

        self.scale = nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=True)

    def forward(self, A, B=1, fixSquare=1, oneSquare=1, mode='one'):
        # A,B :[batch,3,224,224] fixSquare:[batch,9,3,224,224] oneSquare:[batch,3,224,224]
        if mode == 'two':
            # Calculate 3*3 weight matrix
            batchSize = A.size(0)
            feature = self.attentionNet(torch.cat((A, B), 1))
            # print('feature shape: ', feature.shape)
            weight = self.toWeight(feature)  # [batch,3*3]

            weightSquare = weight.view(batchSize, args.Fang * args.Fang, 1, 1, 1)
            weightSquare = weightSquare.expand(batchSize, args.Fang * args.Fang, 3, 84, 84)
            weightSquare = weightSquare * fixSquare  # [batch,9,3,224,224]
            weightSquare = torch.sum(weightSquare, dim=1)  # [batch,3,224,224]

            C = weightSquare * A + (oneSquare - weightSquare) * B
            Cfeature = self.CNet(C)
            return Cfeature, weight, feature

        elif mode == 'one':
            # Calculate feature
            Cfeature = self.CNet(A)
            return Cfeature

        elif mode == 'fc':
            # Go through fc layer, just for debug
            Cfeature = self.fc(A)
            return Cfeature


def euclidean_dist(x, y, model):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    # print('euc:  ', x.shape, y.shape)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    # To accelerate training, but observe little effect
    # A = model.module.scale

    return torch.pow(x - y, 2).sum(2)


def iterateMix(supportImages, supportFeatures, supportBelongs, supportReals, unlblImages, unlblFeatures, unlblReals, model, ways, args):
    '''
        Inputs:
            supportImages ways,shots,3,224,224
        Outputs:
            AImages [ways*shots*(1+augnum),3,224,224]
            BImages [ways*shots*(1+augnum),3,224,224]
            ABelongs: The label in [0,way-1]
            Reals: The label in [0,63] # Just for debug
    '''
    center = supportFeatures.view(ways, args.shots, -1).mean(1)

    # dists = euclidean_dist(galleryFeature,center) # [ways*unNum,ways]
    Num = int(unlblFeatures.size(0) / 10)
    # print(Num)
    with torch.no_grad():
        # print('center shape:  ', center.shape, unlblFeatures[:Num].shape)
        dists = euclidean_dist(unlblFeatures[:Num].cuda(), center, model)
        for i in range(1, 10):
            _end = (i + 1) * Num
            if i == 9:
                _end = unlblFeatures.size(0)
            dist = euclidean_dist(unlblFeatures[i * Num:_end].cuda(), center, model)
            dists = torch.cat((dists, dist), dim=0)

    # print(dists.shape)
    dists = dists.transpose(1, 0)  # [ways,ways*unNum]
    # print(dists.shape)

    AImages = torch.FloatTensor(ways * args.shots * (1 + args.augnum), 3, 84, 84)
    ABelongs = torch.LongTensor(ways * args.shots * (1 + args.augnum), 1)
    Reals = torch.LongTensor(ways * args.shots * (1 + args.augnum), 1)

    BImages = torch.FloatTensor(ways * args.shots * (1 + args.augnum), 3, 84, 84)

    _, bh = torch.topk(dists, args.chooseNum, dim=1, largest=False)

    for i in range(ways):
        for j in range(args.shots):

            AImages[i * args.shots * (1 + args.augnum) + j * (args.augnum + 1) + 0] = supportImages[i * args.shots + j]
            ABelongs[i * args.shots * (1 + args.augnum) + j * (args.augnum + 1) + 0] = supportBelongs[i * args.shots + j]
            Reals[i * args.shots * (1 + args.augnum) + j * (args.augnum + 1) + 0] = supportReals[i * args.shots + j]

            BImages[i * args.shots * (1 + args.augnum) + j * (args.augnum + 1) + 0] = supportImages[i * args.shots + j]

            for k in range(args.augnum):

                p = np.random.randint(0, 2)
                if p == 0:
                    AImages[i * args.shots * (1 + args.augnum) + j * (args.augnum + 1) + 1 + k] = torch.flip(supportImages[i * args.shots + j], [2])
                else:
                    AImages[i * args.shots * (1 + args.augnum) + j * (args.augnum + 1) + 1 + k] = supportImages[i * args.shots + j]
                ABelongs[i * args.shots * (1 + args.augnum) + j * (args.augnum + 1) + 1 + k] = supportBelongs[i * args.shots + j]
                Reals[i * args.shots * (1 + args.augnum) + j * (args.augnum + 1) + 1 + k] = supportReals[i * args.shots + j]

                choose = np.random.randint(0, args.chooseNum)
                # BImages[i*args.shots*(1+args.augnum)+j*(args.augnum+1)+1+k] = image_datasets['test'].get_image(Gallery[bh[i][choose]])

                BImages[i * args.shots * (1 + args.augnum) + j * (args.augnum + 1) + 1 + k] = unlblImages[bh[i][choose]]

                # special test ######
                # c = supportReals[i*args.shots+j]
                # print(c.item())
                # print(unlblReals)
                # idx_same_c = np.where(unlblReals==c.item())[0]
                # print(idx_same_c)
                # idx_Bimages = np.random.choice(idx_same_c, 1, False).item()
                # Bimages[i*args.shots*(1+args.augnum)+j*(args.augnum+1)+1+k] = unlblImages[idx_Bimages]

                ######

                # BImages[i*args.shots*(1+args.augnum)+j*(args.augnum+1)+1+k] = unImages[bh[i][choose]]

    return AImages, BImages, ABelongs, Reals


def batchModel(model, AInputs, requireGrad):
    Batch = (AInputs.size(0) + args.batchSize - 1) // args.batchSize
    First = True
    Cfeatures = 1

    for b in range(Batch):
        if b < Batch - 1:
            midFeature = model(Variable(AInputs[b * args.batchSize:(b + 1) * args.batchSize].cuda(), requires_grad=requireGrad))
        else:
            midFeature = model(Variable(AInputs[b * args.batchSize:AInputs.size(0)].cuda(), requires_grad=requireGrad))

        if First:
            First = False
            Cfeatures = midFeature
        else:
            Cfeatures = torch.cat((Cfeatures, midFeature), dim=0)

    return Cfeatures


def train_model(model, logger, num_epochs=25):
    rootdir = os.getcwd()

    args = Options().parse()

    # To do 1: Change the directory below to the folder where you save miniImagenet pickle files
    ren_data = {x: mini.MiniImagenet("datasplit", x, \
                                     nshot=args.shots, \
                                     nway=args.trainways, \
                                     num_unlabel=args.num_unlabel, \
                                     num_test=args.test_num)
                for x in ['train', 'val', 'test']}

    image_datasets = {}

    image_datasets = {'train': oneShotBaseCls_84.miniImagenetOneshotDataset(ren_data=ren_data['train'], type='train', ways=args.trainways,
                                                                            shots=args.shots, test_num=args.test_num, epoch=1000),
                      'test': oneShotBaseCls_84.miniImagenetOneshotDataset(ren_data=ren_data['test'], type='test', ways=args.ways,
                                                                           shots=args.shots, test_num=args.test_num, epoch=600)}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                  shuffle=(x == 'train'), num_workers=args.nthreads)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    ########################################################
    #### prototypical-network-pytorch
    # from torch.utils.data import DataLoader
    # from mini_imagenet import MiniImageNet
    # from samplers import CategoriesSampler

    # trainset = MiniImageNet('train')
    # train_sampler = CategoriesSampler(trainset.label, 100,
    #                                   5, 1 + 15)
    # train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
    #                           num_workers=8, pin_memory=True)

    # testset = MiniImageNet('test')
    # test_sampler = CategoriesSampler(testset.label, 400,
    #                                   5, 1 + 15)
    # test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
    #                           num_workers=8, pin_memory=True)

    # dataloaders = {'train':train_loader, 'test':test_loader}

    ######################################################################
    # Weight matrix pre-process

    patch_xl = []
    patch_xr = []
    patch_yl = []
    patch_yr = []

    if args.Fang == 3:
        point = [0, 28, 56, 84]
        # point = [0,74,148,224]
    # elif args.Fang == 5:
    #     point = [0,44,88,132,176,224]
    # elif args.Fang == 7:
    #     point = [0,32,64,96,128,160,192,224]

    for i in range(args.Fang):
        for j in range(args.Fang):
            patch_xl.append(point[i])
            patch_xr.append(point[i + 1])
            patch_yl.append(point[j])
            patch_yr.append(point[j + 1])

    fixSquare = torch.zeros(1, args.Fang * args.Fang, 3, 84, 84).float()
    for i in range(args.Fang * args.Fang):
        fixSquare[:, i, :, patch_xl[i]:patch_xr[i], patch_yl[i]:patch_yr[i]] = 1.00
    fixSquare = fixSquare.cuda()

    oneSquare = torch.ones(1, 3, 84, 84).float()
    oneSquare = oneSquare.cuda()
    ######################################################################
    # plot related
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import imshow
    #################################################3

    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000000.0

    detransform = transforms.Compose([
        Denormalize(mu, sigma),
        Clip(),
        transforms.ToPILImage(),
    ])

    #############################################
    # Define the optimizer
    print(args.clsLR)

    if torch.cuda.device_count() > 1:
        if args.scratch == 0:
            optimizer_attention = torch.optim.Adam([
                {'params': model.module.attentionNet.parameters()},
                {'params': model.module.toWeight.parameters(), 'lr': args.LR}
            ], lr=args.LR)  # 0.001
            optimizer_classifier = torch.optim.Adam([
                {'params': model.module.CNet.parameters(), 'lr': args.clsLR},
                {'params': model.module.fc.parameters(), 'lr': args.clsLR}
            ])  # 0.00003
            optimizer_scale = torch.optim.Adam([
                {'params': model.module.scale}
            ], lr=args.LR)  # 0.001
        else:
            optimizer_attention = torch.optim.Adam([
                {'params': model.module.ANet.parameters()},
                {'params': model.module.BNet.parameters()},
                {'params': model.module.toWeight.parameters()}
            ], lr=args.LR)
            optimizer_classifier = torch.optim.Adam([
                {'params': model.module.CNet.parameters()},
                {'params': model.module.fc.parameters()}
            ], lr=args.LR)

    else:
        if args.scratch == 0:
            optimizer_attention = torch.optim.Adam([
                {'params': model.attentionNet.parameters()},
                {'params': model.toWeight.parameters(), 'lr': args.LR}
            ], lr=args.LR)  # 0.001
            optimizer_classifier = torch.optim.Adam([
                {'params': model.CNet.parameters(), 'lr': args.clsLR * 0.1},
                {'params': model.fc.parameters(), 'lr': args.clsLR}
            ])  # 0.00003
            optimizer_scale = torch.optim.Adam([
                {'params': model.scale}
            ], lr=args.LR)  # 0.001
        else:
            optimizer_attention = torch.optim.Adam([
                {'params': model.ANet.parameters()},
                {'params': model.BNet.parameters()},
                {'params': model.toWeight.parameters()}
            ], lr=args.LR)
            optimizer_classifier = torch.optim.Adam([
                {'params': model.CNet.parameters()},
                {'params': model.fc.parameters()}
            ], lr=args.LR)
    # else:
    # optimizer_GNet = torch.optim.Adam([
    # {'params': base_params},
    # {'params': GNet.toWeight.parameters(), 'lr':  args.LR}
    # ], lr=args.LR*0.1)

    # Attention_lr_scheduler = lr_scheduler.StepLR(optimizer_attention, step_size=20, gamma=0.5)
    # Classifier_lr_scheduler = lr_scheduler.StepLR(optimizer_classifier, step_size=20, gamma=0.5)
    Attention_lr_scheduler = lr_scheduler.StepLR(optimizer_attention, step_size=5, gamma=0.5)
    Classifier_lr_scheduler = lr_scheduler.StepLR(optimizer_classifier, step_size=5, gamma=0.5)
    clsCriterion = nn.CrossEntropyLoss()

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        for phase in ['test', 'train']:  ####@@@
            # for phase in [ 'test']: ###
            print('phase {}'.format(phase))

            if phase == 'train':
                Attention_lr_scheduler.step()
                Classifier_lr_scheduler.step()

            model.train(False)  # To ban batchnorm

            running_loss = 0.0
            running_accuracy = 0
            running_cls_loss = 0
            running_cls_accuracy = 0

            Times = 0

            # Iterate over data.
            allWeight = {}
            for k in range(args.Fang * args.Fang):
                allWeight[str(k)] = []

            np.random.seed()

            # for i, batch in tqdm(enumerate(dataloaders[phase], 1)):

            #     data, _ = [_.cuda() for _ in batch]
            #     p = 1 * 5
            #     data_shot, data_query = data[:p], data[p:]

            #     Cfeatures = model(data_shot)
            #     testFeatures = model(data_query)

            #     testLabels = torch.arange(5).repeat(15)
            #     testLabels = testLabels.type(torch.cuda.LongTensor)

            #     Times = Times + 1
            #     ways = 5

            for i, (supportInputs, supportLabels, supportReals, testInputs, testLabels, testReals, unlblInputs, unlblLabels, unlblReals) in tqdm(enumerate(dataloaders[phase])):

                supportInputs = supportInputs.squeeze(0)
                supportLabels = supportLabels.squeeze(0)
                supportReals = supportReals.squeeze(0)

                testInputs = testInputs.squeeze(0)
                testLabels = testLabels.squeeze(0).cuda()

                unlblInputs = unlblInputs.squeeze(0)
                unlblLabels = unlblLabels.squeeze(0)
                unlblReals = unlblReals.squeeze(0)

                # print('what? ', supportReals, unlblLabels,unlblReals)

                # Cfeatures= model(Variable(supportInputs.cuda(),requires_grad=True))
                # testFeatures = model(Variable(testInputs.cuda(),requires_grad=True))

                supportFeatures = batchModel(model, supportInputs, requireGrad=False)
                testFeatures = batchModel(model, testInputs, requireGrad=True)
                unlblFeatures = batchModel(model, unlblInputs, requireGrad=False)

                # print('supportInputs, unlblInputs',supportInputs.shape, unlblInputs.shape)
                # print('supportFeatures, unlblFeatures',supportFeatures.shape, unlblFeatures.shape)

                Times = Times + 1
                ways = 5

                AInputs, BInputs, ABLabels, ABReals = iterateMix(supportInputs, \
                                                                 supportFeatures, \
                                                                 supportLabels, \
                                                                 supportReals, \
                                                                 unlblInputs, \
                                                                 unlblFeatures, \
                                                                 unlblReals, \
                                                                 model=model, \
                                                                 ways=ways, \
                                                                 args=args)

                Batch = (AInputs.size(0) + args.batchSize - 1) // args.batchSize

                First = True
                Cfeatures = 1
                Ccls = 1
                Weights = 0

                # print('Ainputs, Binputs',AInputs.shape, BInputs.shape)

                '''
                    Pytorch has a bug.
                    Per input's size has to be divisble by the number of GPU
                    So make sure each input's size can be devisble by the number of available GPU
                '''

                for b in range(Batch):
                    # print('b: ', b)
                    if b < Batch - 1:
                        # _cfeature= model(Variable(AInputs[b*args.batchSize:(b+1)*args.batchSize].cuda(),requires_grad=True))

                        _cfeature, weight, middleFeature = model(Variable(AInputs[b * args.batchSize:(b + 1) * args.batchSize].cuda(), requires_grad=True),
                                                                 Variable(BInputs[b * args.batchSize:(b + 1) * args.batchSize].cuda(), requires_grad=True),
                                                                 Variable(fixSquare.expand(args.batchSize, args.Fang * args.Fang, 3, 84, 84).cuda(), requires_grad=False),
                                                                 Variable(oneSquare.expand(args.batchSize, 3, 84, 84).cuda(), requires_grad=False),
                                                                 mode='two'
                                                                 )

                        _cls = model(_cfeature, B=1, fixSquare=1, oneSquare=1, mode='fc')
                    else:
                        _len = AInputs.size(0) - (b * args.batchSize)
                        # _cfeature= model(Variable(AInputs[b*args.batchSize:].cuda(),requires_grad=True))

                        _cfeature, weight, middleFeature = model(Variable(AInputs[b * args.batchSize:].cuda(), requires_grad=True),
                                                                 B=Variable(BInputs[b * args.batchSize:].cuda(), requires_grad=True),
                                                                 fixSquare=Variable(fixSquare.expand(_len, args.Fang * args.Fang, 3, 84, 84).cuda(), requires_grad=False),
                                                                 oneSquare=Variable(oneSquare.expand(_len, 3, 84, 84).cuda(), requires_grad=False),
                                                                 mode='two'
                                                                 )
                        _cls = model(_cfeature, B=1, fixSquare=1, oneSquare=1, mode='fc')

                    if First:
                        First = False
                        Cfeatures = _cfeature
                        # print('First', Cfeatures.shape)
                        Weights = weight
                        Ccls = _cls
                    else:
                        Cfeatures = torch.cat((Cfeatures, _cfeature), dim=0)
                        # print('Not First', Cfeatures.shape)
                        Weights = torch.cat((Weights, weight), dim=0)
                        Ccls = torch.cat((Ccls, _cls), dim=0)

                # for i,(supportInputs,supportLabels,supportReals,testInputs,testLabels,testReals,unlblInputs,unlblLabels,unlblReals) in tqdm(enumerate(dataloaders[phase])):

                # if epoch == 0 and i>4000:
                #     break

                # Times = Times + 1

                # supportInputs = supportInputs.squeeze(0)
                # supportLabels = supportLabels.squeeze(0)
                # supportReals = supportReals.squeeze(0)

                # testInputs = testInputs.squeeze(0)
                # testLabels = testLabels.squeeze(0).cuda()

                # # unlblInputs = unlblInputs.squeeze(0)
                # # unlblLabels = unlblLabels.squeeze(0)
                # # unlblReals = unlblReals.squeeze(0)

                # ways = int(supportInputs.size(0)/args.shots)

                # # supportFeatures = batchModel(model,supportInputs,requireGrad=False)
                # testFeatures = batchModel(model,testInputs,requireGrad=True)
                # # unlblFeatures = batchModel(model,unlblInputs,requireGrad=False)

                # AInputs, ABLabels, ABReals = supportInputs, supportLabels, supportReals

                # Batch = (AInputs.size(0)+args.batchSize-1)//args.batchSize

                # First = True
                # Cfeatures = 1
                # Ccls = 1
                # Weights = 0

                # '''
                #     Pytorch has a bug.
                #     Per input's size has to be divisble by the number of GPU
                #     So make sure each input's size can be devisble by the number of available GPU
                # '''

                # for b in range(Batch):
                #     if b<Batch-1:
                #         _cfeature= model(Variable(AInputs[b*args.batchSize:(b+1)*args.batchSize].cuda(),requires_grad=True))

                #         # _cfeature, weight, middleFeature = model(Variable(AInputs[b*args.batchSize:(b+1)*args.batchSize].cuda(),requires_grad=True),
                #         #     Variable(BInputs[b*args.batchSize:(b+1)*args.batchSize].cuda(),requires_grad=True),
                #         #     Variable(fixSquare.expand(args.batchSize,args.Fang*args.Fang,3,224,224).cuda(),requires_grad=False),
                #         #     Variable(oneSquare.expand(args.batchSize,3,224,224).cuda(),requires_grad=False),
                #         #     mode='two'
                #         #     )
                #         #_cls = model(_cfeature,B=1,fixSquare=1,oneSquare=1,mode='fc')
                #     else:
                #         _len = AInputs.size(0)-(b*args.batchSize)
                #         _cfeature= model(Variable(AInputs[b*args.batchSize:].cuda(),requires_grad=True))

                #         # _cfeature, weight, middleFeature = model(Variable(AInputs[b*args.batchSize:].cuda(),requires_grad=True),
                #         #     B=Variable(BInputs[b*args.batchSize:].cuda(),requires_grad=True),
                #         #     fixSquare=Variable(fixSquare.expand(_len,args.Fang*args.Fang,3,224,224).cuda(),requires_grad=False),
                #         #     oneSquare=Variable(oneSquare.expand(_len,3,224,224).cuda(),requires_grad=False),
                #         #     mode='two'
                #         #     )
                #         #_cls = model(_cfeature,B=1,fixSquare=1,oneSquare=1,mode='fc')

                #     if First:
                #         First = False
                #         Cfeatures = _cfeature
                #         # Weights = weight
                #         #Ccls = _cls
                #     else:
                #         Cfeatures = torch.cat((Cfeatures,_cfeature),dim=0)
                #         # Weights = torch.cat((Weights,weight),dim=0)
                #         #Ccls = torch.cat((Ccls,_cls),dim=0)

                Weights = Weights.transpose(1, 0)  # 9*Batch

                for k in range(args.Fang * args.Fang):
                    allWeight[str(k)] = allWeight[str(k)] + Weights[k].view(-1).tolist()

                if args.augnum == 0:
                    center = Cfeatures.view(ways, args.shots * (1), -1).mean(1)
                else:
                    center = Cfeatures.view(ways, args.shots * (1 + args.augnum), -1).mean(1)

                # center = Cfeatures.view(ways,args.shots*(1+args.augnum),-1).mean(1) # [ways,512]
                dists = euclidean_dist(testFeatures, center, model)  # [ways*test_num,ways]

                log_p_y = F.log_softmax(-dists, dim=1).view(ways, args.test_num, -1)  # [ways,test_num,ways]

                loss_val = -log_p_y.gather(2, testLabels.view(ways, args.test_num, 1)).squeeze().view(-1).mean()

                _, y_hat = log_p_y.max(2)

                acc_val = torch.eq(y_hat, testLabels.view(ways, args.test_num)).float().mean()

                # statistics
                running_loss += loss_val.item()
                running_accuracy += acc_val.item()

                # backward + optimize only if in training phase

                if phase == 'train':
                    if (args.fixAttention == 0):
                        optimizer_attention.zero_grad()
                        loss_val.backward(retain_graph=True)
                        optimizer_attention.step()
                    # if args.fixScale == 0:
                    #     optimizer_scale.zero_grad()
                    #     loss_val.backward(retain_graph=True)
                    #     optimizer_scale.step()
                    _, preds = torch.max(Ccls, 1)
                    ABReals = ABReals.view(ABReals.size(0)).cuda()
                    loss_cls = clsCriterion(Ccls, ABReals)
                    if (args.fixCls == 0):
                        optimizer_classifier.zero_grad()
                        loss_val.backward(retain_graph=True)
                        # loss_cls.backward()
                        optimizer_classifier.step()

                    running_cls_loss += loss_cls.item()
                    running_cls_accuracy += torch.eq(preds, ABReals).float().mean()

                    # Attention_lr_scheduler.step()
                    # Classifier_lr_scheduler.step()

            epoch_loss = running_loss / (Times * 1.0)
            epoch_accuracy = running_accuracy / (Times * 1.0)
            epoch_cls_loss = running_cls_loss / (Times * 1.0)
            epoch_cls_accuracy = running_cls_accuracy / (Times * 1.0)

            # info = {
            #     phase+'loss': epoch_loss,
            #     phase+'accuracy': epoch_accuracy,
            #     phase+'_cls_loss': epoch_cls_loss,
            #     phase+'_cls_accuracy': epoch_cls_accuracy,
            # }

            # Logging
            logger.info("=========================================")
            logger.info(f"Epoch {epoch}, Phase {phase}")
            logger.info("=========================================")
            logger.info(f"loss: {epoch_loss:.4f};")
            logger.info(f"accuracy: {epoch_accuracy:.4f};")
            logger.info(f"_cls_loss: {epoch_cls_loss:.4f};")
            logger.info(f"_cls_accuracy: {epoch_cls_accuracy:.4f};")

            # for tag, value in info.items():
            #    logger.scalar_summary(tag, value, epoch+1)

            # print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
            #     phase, epoch_loss,epoch_accuracy))

            # print('Classify Loss: {:.4f} Accuracy: {:.4f}'.format(
            #     epoch_cls_loss,epoch_cls_accuracy))

            # deep copy the model
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss

                if torch.cuda.device_count() > 1:
                    best_model_wts = copy.deepcopy(model.module.state_dict())
                else:
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()
        if epoch % 2 == 0:
            torch.save(best_model_wts, os.path.join(rootdir, 'models/' + str(args.tensorname) + '.t7'))
            print('save!')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# .. to load your previously training model:
# model.load_state_dict(torch.load('mytraining.pt'))
def run():
    logger = get_logger(name=args.name_log)
    model = GNet()

    if args.GNet != 'none':
        model.load_state_dict(torch.load('models/' + args.GNet + '.t7', map_location=lambda storage, loc: storage))
        print('loading ', args.GNet)

    # if torch.cuda.device_count() > 1:
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.cuda()

    model = train_model(model, logger=logger, num_epochs=120)
    ##

    # ... after training, save your model 

    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(rootdir, 'models/' + str(args.tensorname) + '.t7'))
    else:
        torch.save(model.state_dict(), os.path.join(rootdir, 'models/' + str(args.tensorname) + '.t7'))


if __name__ == '__main__':
    run()
