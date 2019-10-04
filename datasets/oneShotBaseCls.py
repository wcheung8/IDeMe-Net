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
userName = getpass.getuser()
import pickle as pkl
pathminiImageNet = 'C://Users/bezer/Desktop/IDeMe-Net/'
pathImages = os.path.join(pathminiImageNet,'images/')
# LAMBDA FUNCTIONS

np.random.seed(2191)

patch_xl = [0,0,0,74,74,74,148,148,148]
patch_xr = [74,74,74,148,148,148,224,224,224]
patch_yl = [0,74,148,0,74,148,0,74,148]
patch_yr = [74,148,224,74,148,224,74,148,224]

class miniImagenetOneshotDataset(data.Dataset):
    def __init__(self, dataroot = 'C://Users/bezer/Desktop/IDeMe-Net/datasplit', type = 'train',ways=5,shots=1,test_num=1,epoch=100,galleryNum = 10):
        # oneShot setting
        self.ways = ways
        self.shots = shots
        self.test_num = test_num # indicate test number of each class
        self.__size = epoch

        self.transform = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])

        self.galleryTransform = transforms.Compose([
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])


        def _read_cache(cache_path):
            dictLabels = {}
            c = 0
            if os.path.exists(cache_path):
              try:
                with open(cache_path, "rb") as f:
                    data = pkl.load(f, encoding='bytes')
                  
                    for img, cls in zip(data[b'image_data'], data[b'class_dict']):
                        if cls in dictLabels.keys():
                            dictLabels[cls].append(img)
                        else:       
                            dictLabels[cls] = [img]
                return dictLabels
              except:
                with open(cache_path, "rb") as f:
                    data = pkl.load(f)
                    
                    for cls, idxs in data['class_dict'].items():
                    
                        dictLabels[cls] = []
                        
                        for i in idxs:
                            dictLabels[cls].append(Image.fromarray(data['image_data'][i]))
                return dictLabels
            else:
                print("NO FILE FOUND")
                exit()

    
        self.miniImagenetImagesDir = os.path.join(dataroot,'images')

        self.unData = _read_cache(cache_path = os.path.join(dataroot,'mini-imagenet-cache-train' + '.pkl'))
        self.data = _read_cache(cache_path   = os.path.join(dataroot,'mini-imagenet-cache-' + type + '.pkl'))

        self.type = type    
        self.data = collections.OrderedDict(sorted(self.data.items()))
        self.unData = collections.OrderedDict(sorted(self.unData.items()))
        self.galleryNum = galleryNum    

        # sample Gallery
        self.Gallery = []
        import random
        random.seed(2019)
        for classes in self.unData.keys():
            Files = random.sample(self.unData[classes], self.galleryNum)
            for file in Files:
                self.Gallery.append(file)
    
        numpy.random.seed()

        self.keyTobh = {}
        for c in range(len(self.data.keys())):
            self.keyTobh[list(self.data.keys())[c]] = c

        for c in range(len(self.unData.keys())):
            self.keyTobh[list(self.unData.keys())[c]] = c

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
                image = self.transform(self.Gallery[j])
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

    def get_image(self,img):
        image = self.galleryTransform(img)
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
        


