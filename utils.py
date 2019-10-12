import argparse
from torch.utils.data import DataLoader
import csv
import cv2
import numpy as np
import os
import pickle as pkl
import torch
from torch.utils.data import Dataset 
from torchvision import transforms


class AttrDict(dict):

     def __init__(self):
           self.__dict__ = self


args = AttrDict()

args.disable_distractor = True
args.label_ratio = None

# args = argparse.ArgumentParser()

# # args.add_argument('--data_dir',
# #                   type = str,
# #                   default = './data/',
# #                   help = "dir where training is conducted")
# # args.add_argument('--logs_train_dir',
# #                   type = str,
# #                   default = './logs/loss_record/',
# #                   help = "dir where summary is saved")                 
# # args.add_argument('--testset_name',
# #                   type = str,
# #                   default = 'testset',
# #                   help = "the test set where the test is conducted")
# args.add_argument('--disable_distractor',
#                   type = bool,
#                   default = True,
#                   help = "whether use distractor in the training")
# args.add_argument('--label_ratio',
#                   type = float,
#                   default = None,
#                   help = "Ratio of labeled data")
# # args.add_argument('--seq_start',
# #                   type = int,
# #                   default = 5,
# #                   help = """start of the sequence generation""")
# # args.add_argument('--factor',
# #                   type = float,
# #                   default = .0005,
# #                   help = """factor of regularization""")


# args = args.parse_args()
