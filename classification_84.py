import os
import numpy as np
import argparse
import torch
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
import matplotlib.pyplot as plt
from option import Options
from datasets import softRandom_84 as softRandom
from torch.optim import lr_scheduler
import copy
import time
import math
rootdir = os.getcwd()

args = Options().parse()

from datasets import miniimagenet as mini


# To do 1: Change the directory below to the folder where you save miniImagenet pickle files
ren_data = {x: mini.MiniImagenet("/home/root/data/miniImagenet", x, nshot=240, num_distractor=0, nway=64) for x in ['train']}

all_labeled = ren_data['train'].__getAllLabeled__()


image_datasets = {x: softRandom.miniImagenetEmbeddingDataset(ren_data=all_labeled,type=x)
                  for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batchSize,
                                             shuffle=True, num_workers=args.nthreads)
              for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}

print('datasize: ', dataset_sizes)


######################################################################
# Define the Embedding Network
# resnet18 without fc layer
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        if self.maml:
            self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN     = BatchNorm2d_fw(outdim)
        else:
            self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
            self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, flatten = True):
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self,x):
        out = self.trunk(x)
        return out

def Conv4():
    return ConvNet(4)


def Conv6():
    return ConvNet(6)
##############################




class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()
        self.convnet = Conv6()
        num_ftrs = self.convnet.final_feat_dim
        self.convnet.fc = nn.Linear(num_ftrs,64)

    def forward(self,inputs):
        outputs = self.convnet(inputs)
        
        return outputs

classificationNetwork = ClassificationNetwork()
if args.network!='None':
    classificationNetwork.load_state_dict(torch.load('models/'+str(args.network)+'.t7', map_location=lambda storage, loc: storage))
    print('loading ',str(args.network))
classificationNetwork = classificationNetwork.cuda()

my_list = ['convnet.fc.weight', 'convnet.fc.bias']
params = list(filter(lambda kv: kv[0] in my_list, classificationNetwork.named_parameters()))
base_params = list(filter(lambda kv: kv[0] not in my_list, classificationNetwork.named_parameters()))##

# print(params,base_params)

#############################################
#Define the optimizer#

criterion = nn.CrossEntropyLoss()

if args.network=='None':
    optimizer_embedding = optim.Adam([
                    {'params': classificationNetwork.parameters()},
                ], lr=0.001)
else:
    optimizer_embedding = optim.Adam([
                    {'params': params,'lr': args.LR*0.1},
                    {'params': base_params, 'lr': args.LR}##
                ])

embedding_lr_scheduler = lr_scheduler.StepLR(optimizer_embedding, step_size=10, gamma=0.5)


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [ 'train']:
            
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
                model.eval()
            

            running_loss = 0.0 
            tot_dist = 0.0
            running_corrects = 0
            loss = 0

            inputsSum = 0
            labelsSum = 0

            # Iterate over data.
            for i,(inputs,labels) in tqdm(enumerate(dataloaders[phase])):

                inputsSum += torch.mean(inputs.data.float())
                labelsSum += torch.mean(labels.data.float())

                #c = labels
                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                _, preds = torch.max(outputs, 1)

                labels = labels.view(labels.size(0))

                loss = criterion(outputs, labels)


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.view(-1)).item()
                #print(running_corrects)


            print('sums', phase, inputsSum, labelsSum)

            print(running_corrects)


            epoch_loss = running_loss / (dataset_sizes[phase]*1.0)
            epoch_acc = running_corrects / (dataset_sizes[phase]*1.0)
            info = {
                phase+'loss': running_loss,
                phase+'Accuracy': epoch_acc,
            }

            print('{} Loss: {:.4f} Accuracy: {:.4f} '.format(
                phase, epoch_loss,epoch_acc))

            # deep copy the model
            if phase == 'train' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        
        print()
        # if epoch>=30 and epoch %3 ==0:
        #     torch.save(best_model_wts,os.path.join(rootdir,'models/'+str(args.tensorname)+ str(epoch) + '.t7'))
        #     print('save!')
        if epoch % 10 ==0:
            torch.save(best_model_wts,os.path.join(rootdir,'models/'+str(args.tensorname)+ '.t7'))
            print('save!')
        ##

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


classificationNetwork = train_model(classificationNetwork, criterion, optimizer_embedding,
                         embedding_lr_scheduler, num_epochs=25)##


torch.save(classificationNetwork.state_dict(),os.path.join(rootdir,'models/'+str(args.tensorname)+'.t7'))


