#!/bin/bash
wait
python3 train.py --model Swin --pretrain 1 --img_size 384 -bs 64 -ep 1 --loss FL --lr 3e-4 --fold 1 --device 0 1
wait
python3 train.py --model Swin --pretrain 1 --img_size 384 -bs 64 -ep 1 --loss FL --lr 3e-4 --fold 2 --device 0 1
wait
python3 train.py --model Swin --pretrain 1 --img_size 384 -bs 64 -ep 1 --loss FL --lr 3e-4 --fold 3 --device 0 1
wait
python3 train.py --model Swin --pretrain 1 --img_size 384 -bs 64 -ep 1 --loss FL --lr 3e-4 --fold 4 --device 0 1
wait
python3 train.py --model Swin --pretrain 1 --img_size 384 -bs 64 -ep 1 --loss FL --lr 3e-4 --fold 5 --device 0 1