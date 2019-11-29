import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import os.path
import csv
import math
import collections
from tqdm import tqdm
import datetime

import numpy as np
import numpy
#from watch import NlabelTovector
import getpass  
from option import Options


args = Options().parse()


userName = getpass.getuser()

if args.dataset == 'miniImageNet':
    pathminiImageNet = '/home/'+userName+'/data/miniImagenet/'
elif args.dataset == 'CUB':
    pathminiImageNet = '/home/'+userName+ '/data/CUB/CUB_200_2011'

pathImages = os.path.join(pathminiImageNet,'images/')
# LAMBDA FUNCTIONS
filenameToPILImage = lambda x: Image.open(x)

np.random.seed(2191)

patch_xl = [0,0,0,74,74,74,148,148,148]
patch_xr = [74,74,74,148,148,148,224,224,224]
patch_yl = [0,74,148,0,74,148,0,74,148]
patch_yr = [74,148,224,74,148,224,74,148,224]

class miniImagenetOneshotDataset(data.Dataset):
    def __init__(self, dataroot = '/home/'+userName+'/data/miniImagenet', type = 'train',ways=5,shots=1,test_num=1,epoch=100,galleryNum = 10):
        

        if args.dataset == 'miniImageNet':
            dataroot = '/home/'+userName+'/data/miniImagenet/'
        elif args.dataset == 'CUB':
            dataroot = '/home/'+userName+ '/data/CUB/CUB_200_2011'


        # oneShot setting
        self.ways = ways
        self.shots = shots
        self.test_num = test_num # indicate test number of each class
        self.__size = epoch


        # with a inputSize of 84

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                                ])

        self.galleryTransform = transforms.Compose([
                                            filenameToPILImage,
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Resize(84),
                                            transforms.CenterCrop(84),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                            ])
        # with a inputSize of 224

        # self.transform = transforms.Compose([filenameToPILImage,
        #                                 transforms.Resize(256),
        #                                 transforms.CenterCrop(224),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #                                 ])

        # self.galleryTransform = transforms.Compose([filenameToPILImage,
        #                                     transforms.RandomHorizontalFlip(),
        #                                     transforms.Resize(256),
        #                                     transforms.CenterCrop(224),
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #                                     ])





        def loadSplit(splitFile):
            dictLabels = {}
            with open(splitFile) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i,row in enumerate(csvreader):
                    filename = row[0]
                    label = row[1]

                    if label in dictLabels.keys():
                        dictLabels[label].append(filename)
                    else:
                        dictLabels[label] = [filename]
            return dictLabels

        self.miniImagenetImagesDir = os.path.join(dataroot,'images')

        self.unData = loadSplit(splitFile = os.path.join(dataroot,'train' + '.csv'))
        self.data = loadSplit(splitFile = os.path.join(dataroot,type + '.csv'))

        self.type = type
        self.data = collections.OrderedDict(sorted(self.data.items()))
        self.unData = collections.OrderedDict(sorted(self.unData.items()))
        self.galleryNum = galleryNum

        # sample Gallery
        self.Gallery = []
        numpy.random.seed(2019)
        for classes in range(len(self.unData.keys())):
            Files = np.random.choice(self.unData[self.unData.keys()[classes]], self.galleryNum, False)
            for file in Files:
                self.Gallery.append(file)

        numpy.random.seed()

        self.keyTobh = {}
        for c in range(len(self.data.keys())):
            self.keyTobh[self.data.keys()[c]] = c

        for c in range(len(self.unData.keys())):
            self.keyTobh[self.unData.keys()[c]] = c

        #print(self.keyTobh)
    def batchModel(model,AInputs,requireGrad):
        Batch = (AInputs.size(0)+args.batchSize-1)//args.batchSize
        First = True
        Cfeatures = 1


        for b in range(Batch):
            if b<Batch-1:
                midFeature = model(Variable(AInputs[b*args.batchSize:(b+1)*args.batchSize].cuda(),requires_grad=requireGrad))
            else:
                midFeature = model(Variable(AInputs[b*args.batchSize:AInputs.size(0)].cuda(),requires_grad=requireGrad))

            if First:
                First = False
                Cfeatures = midFeature
            else:
                Cfeatures = torch.cat((Cfeatures,midFeature),dim=0)

        return Cfeatures

    def acquireFeature(self,model,batchSize=128):

        Batch = (len(self.Gallery)+batchSize-1)//batchSize
        First = True
        Cfeatures = 1
        Images = 1

        for b in range(Batch):

            jFirst = True
            Images = 1
            for j in range(b*batchSize,min((b+1)*batchSize,len(self.Gallery))):
                image = self.transform(os.path.join(pathImages,str(self.Gallery[j])))
                image = image.unsqueeze(0)
                if jFirst:
                    jFirst=False
                    Images = image
                else:
                    Images = torch.cat((Images,image),0)

            with torch.no_grad():##
                midFeature = model(Variable(Images.cuda(),requires_grad=False)).cpu()
               
                if First:
                    First = False
                    Cfeatures = midFeature
                else:
                    Cfeatures = torch.cat((Cfeatures,midFeature),dim=0)

        return Cfeatures

    def get_image(self,file):
        image = self.galleryTransform(os.path.join(pathImages,str(file)))
        return image


    def __getitem__(self, index):
        # ways,shots,3,224,224
        #numpy.random.seed(index+datetime.datetime.now().second + datetime.datetime.now().microsecond)
        supportFirst = True
        supportImages = 1
        supportBelongs = torch.LongTensor(self.ways*self.shots,1)
        supportReal = torch.LongTensor(self.ways*self.shots,1)

        testFirst = True
        testImages = 1
        testBelongs = torch.LongTensor(self.ways*self.test_num,1)
        testReal = torch.LongTensor(self.ways*self.test_num,1)

        selected_classes = np.random.choice(self.data.keys(), self.ways, False)
        for i in range(self.ways):
            files = np.random.choice(self.data[selected_classes[i]], self.shots, False)
            for j in range(self.shots):
                image = self.transform(os.path.join(pathImages,str(files[j])))
                image = image.unsqueeze(0)
                if supportFirst:
                    supportFirst=False
                    supportImages = image
                else:
                    supportImages = torch.cat((supportImages,image),0)
                supportBelongs[i*self.shots+j,0] = i
                supportReal[i*self.shots+j,0] = self.keyTobh[selected_classes[i]]


            files = np.random.choice(self.data[selected_classes[i]], self.test_num, False)
            for j in range(self.test_num):
                image = self.transform(os.path.join(pathImages,str(files[j])))
                image = image.unsqueeze(0)
                if testFirst:
                    testFirst = False
                    testImages = image
                else:
                    testImages = torch.cat((testImages,image),0)
                testBelongs[i*self.test_num+j,0] = i
                testReal[i*self.test_num+j,0] = self.keyTobh[selected_classes[i]]


        return supportImages,supportBelongs,supportReal,testImages,testBelongs,testReal

    def __len__(self):
        return self.__size

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# if __name__ == '__main__':
#     dataTrain = torch.utils.data.DataLoader(miniImagenetOneshotDataset(type='train',ways=5,shots=5,test_num=15,epoch=1000),batch_size=1,shuffle=False,num_workers=2,worker_init_fn=worker_init_fn)


#     for j in range(5):
#         np.random.seed()
#         print('\n\n\n\n')
#         for i,(supportInputs,supportBelongs,supportReals,testInputs,testBelongs,testReals,unInputs,unBelongs,unReals) in tqdm(enumerate(dataTrain)):
#             pass
#             # 
#             #if i<3:
#             #    print(supportInputs[0][0][0])
            
#             # haha = 1
#             # if i<=5:
#             #     print(i,supportInputs.size(),supportBelongs.size(),testInputs.size(),testBelongs.size())
#             #print(testLabels)
#             if i<2:
#                 # print(supportBelongs,supportReals)
#                 print(supportReals)
#             else:
#                 break
