#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python classification_84.py --tensorname conv6
for i in 1 2 3 4
do
   # CUDA_VISIBLE_DEVICES=0 python onlyBasetwoLoss_84.py --network conv6 --shots 5 --augnum 5 --fixCls 1 --tensorname convNet-layer$i-5shot --chooseNum 30 --mixupLayer $i
   CUDA_VISIBLE_DEVICES=0 python onlyBasetwoLoss_84.py --network conv6 --shots 5 --augnum 5 --fixCls 0 --fixAttention 1 --GNet convNet-layer$i-5shot --tensorname convNet-layer$i-5shot_round2 --chooseNum 30 --mixupLayer $i
   
done