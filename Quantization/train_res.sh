#!/usr/bin/env bash

################ ResNet20 ################
#32-bit training with cifar10
python main.py --arch res20 --dataset cifar10 --epochs 200 --w_bit 32 -id 0,1 --wd 1e-4

#2-bit APOT Training with cifar10
python main.py --arch res20 --dataset cifar10 --epochs 200 --w_bit 2 --a_bit 2 -id 0,1 --
wd 1e-4 --lr 4e-2 --N_grid False --init result/cifar10/res20_32bit/model_best.pth.tar

#2-bit Our grid Training with cifar10
python main.py --arch res20 --dataset cifar10 --epochs 200 --w_bit 2 --a_bit 2 -id 0,1 --
wd 1e-4 --lr 4e-2 --N_grid True --constant 2 --init result/cifar10/res20_32bit/model_best.pth.tar


################ ResNet32 and ResNet56 and ResNet110################
#############ResNet56 or ResNet110 == --arch res56, 110#############
#32-bit training with cifar100
python main.py --arch res56 --dataset cifar100 --w_bit 32 -id 0,1 --wd 1e-4

#2-bit APOT Training with cifar100
python main.py --arch res56 --dataset cifar100 --w_bit 2 --a_bit 2 -id 0,1 --
wd 1e-4 --lr 4e-2 --N_grid False --init result/cifar100/res56_32bit/model_best.pth.tar

#2-bit Our grid Training with cifar100
python main.py --arch res56 --dataset cifar100 --w_bit 2 --a_bit 2 -id 0,1 --
wd 1e-4 --lr 4e-2 --N_grid True --constant 2 --init result/cifar100/res56_32bit/model_best.pth.tar


#If you should use cifar10 dataset, you can write --dataset cifar10 