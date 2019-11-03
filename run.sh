#!/bin/bash
for i in 1 2 3 4 5
do
   CUDA_VISIBLE_DEVICES=0 python onlyBasetwoLoss_84.py --network conv6 --shots 5 --augnum 5 --fixCls 1 --tensorname convNet-layer1-5shot --chooseNum 30 --mixupLayer $i
done