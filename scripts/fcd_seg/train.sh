#!/usr/bin/env bash

## run the training
python train.py \
--dataroot datasets/fcd_seg \
--name fcd_seg \
--arch meshunet \
--dataset_mode segmentation \
--ncf 16 32 64 \
--ninput_edges 17000 \
--pool_res 10000 6000  \
--resblocks 1 \
--batch_size 1 \
--lr 0.003 \
--num_aug 20 \
--slide_verts 0.2