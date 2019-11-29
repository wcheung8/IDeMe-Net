import glob
import os
import pandas as pd

import json
import cv2

from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

save_to = '/home/root/data/CUB/CUB_200_2011/images'


for split in ['base', 'val', 'novel']:
    
    img_dict = {'filename':[],'label':[]}
    
    with open('filelists/CUB/' + split +'.json') as json_file:
        data = json.load(json_file)
        for i, im in tqdm(enumerate(data["image_names"])):
            #print(im)

            image = transforms.ToTensor()(Image.open(im))
            
            if image.shape[0] == 3:

                filename = im.split('/')[9]
                label = im.split('/')[8]

                #print(img_file, label)

                img_dict['filename'] = img_dict['filename'] + [filename]
                img_dict['label'] = img_dict['label'] + [label]
                
                #print(filename)
                image = transforms.Resize((84))(transforms.ToPILImage()(image))
                image = image.save(save_to+'/'+filename, quality=95) 
            else:
                print(im)
                
            #break
        
        df = pd.DataFrame.from_dict(img_dict)

        print(df.shape)
        
        #df.to_csv(split + '.csv',index=False)
        

