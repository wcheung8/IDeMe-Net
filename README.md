# Learn to hallucinate from paired features for Few-shot Classification

```
If using google cloud, please follow googleCloud_setup_py27.txt to set up your environment

Please first put the data in:
/home/yourusername/data/miniImagenet

The images are put in 
.../miniImagenet/images
such as:miniImagenet\images\n0153282900000006.jpg
We provide the data split in ./datasplit/,please put them at 
.../miniImagenet/train.csv
.../miniImagenet/test.csv
.../miniImagenet/val.csv




# For the first step, we fix the deformation sub-network and train the embedding sub-network with randomly deformed images

# For 224x224
# run python classification.py --tensorname yournetworkname
# For 84x84
# run python classification_84.py --tensorname yournetworkname


# Then, we fix the embedding sub-network and learn the deformation sub-network 

# For 224x224
python onlyBasetwoLoss.py --network pretrainedClassifier --shots 5 --augnum 5 --fixCls 1 --tensorname metaNet_5shot --chooseNum 30 
# For 84x84
python onlyBasetwoLoss_84.py --network pretrainedClassifier_84 --shots 5 --augnum 5 --fixCls 1 --tensorname metaNet_5shot --chooseNum 30 

########################################
#(command below has not been tested yet.)
########################################
# If you want to further improve, then fix one sub-network and iteratively train the other. 

# update cls
python onlyBasetwoLoss.py --network softRandom --shots 5 --augnum 5 --fixCls 0 --fixAttention 1 --tensorname metaNet_5shot_round2 --chooseNum 30 --GNet metaNet_5shot 

We also provide our model: metaNet_1shot.t7 and metaNet_5shot.t7 in ./models

You can use --GNet metaNet_1shot to load the model.


```

> [**Image Deformation Meta-Networks for One-Shot Learning**](<https://arxiv.org/abs/1905.11641>),            
> Zitian Chen, Yanwei Fu, Yu-Xiong Wang, Lin Ma, Wei Liu, Martial Hebert

# ![](picture/meta_learning.png)



![](picture/deformed_images.png)



![](picture/approach.png)



## Installation

```
python=2.7
pytorch=1.0
```

## Datasets

The data split is from [**Semantic Feature Augmentation in Few-shot Learning**](<https://github.com/tankche1/Semantic-Feature-Augmentation-in-Few-shot-Learning>)

```
Please put the data in:
/home/yourusername/data/miniImagenet

The images are put in 
.../miniImagenet/images
such as:miniImagenet\images\n0153282900000006.jpg
We provide the data split in ./datasplit/,please put them at 
.../miniImagenet/train.csv
.../miniImagenet/test.csv
.../miniImagenet/val.csv
```
