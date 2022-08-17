#!/usr/bin/env bash

################ ResNet20 ################
#32-bit training with cifar100
python main.py --arch res20 --dataset cifar100 --epochs 200 --w_bit 32 --a_bit 32 -id 0,1 --wd 1e-4

#2-bit APOT Training with cifar100
python main.py --arch res20 --dataset cifar100 --epochs 200 --w_bit 2 --a_bit 2 -id 0,1 --
wd 1e-4 --lr 4e-2 --N_grid False --init result/cifar100/res20_32bit/model_best.pth.tar

#2-bit Our grid Training with cifar100
python main.py --arch res20 --dataset cifar100 --epochs 200 --w_bit 2 --a_bit 2 -id 0,1 --
wd 1e-4 --lr 4e-2 --N_grid True --init result/cifar100/res20_32bit/model_best.pth.tar


################ ResNet32 ################
#32-bit training with cifar100
python main.py --arch res32 --dataset cifar100 --w_bit 32 --a_bit 32 -id 0,1 --wd 1e-4

#2-bit APOT Training with cifar100
python main.py --arch res32 --dataset cifar100 --w_bit 2 --a_bit 2 -id 0,1 --
wd 1e-4 --lr 4e-2 --N_grid False --init result/cifar100/res32_32bit/model_best.pth.tar

#2-bit Our grid Training with cifar100
python main.py --arch res32 --dataset cifar100 --w_bit 2 --a_bit 2 -id 0,1 --
wd 1e-4 --lr 4e-2 --N_grid True --init result/cifar100/res32_32bit/model_best.pth.tar


################ ResNet56 ################
#32-bit training with cifar100
python main.py --arch res56 --dataset cifar100 --w_bit 32 --a_bit 32 -id 0,1 --wd 1e-4

#2-bit APOT Training with cifar100
python main.py --arch res56 --dataset cifar100 --w_bit 2 --a_bit 2 -id 0,1 --
wd 1e-4 --lr 4e-2 --N_grid False --init result/cifar100/res56_32bit/model_best.pth.tar

#2-bit Our grid Training with cifar100
python main.py --arch res56 --dataset cifar100 --w_bit 2 --a_bit 2 -id 0,1 --
wd 1e-4 --lr 4e-2 --N_grid True --init result/cifar100/res56_32bit/model_best.pth.tar


################ ResNet110 ################
#32-bit training with cifar100
python main.py --arch res110 --dataset cifar100 --w_bit 32 --a_bit 32 -id 0,1 --wd 1e-4

#2-bit APOT Training with cifar100
python main.py --arch res110 --dataset cifar100 --w_bit 2 --a_bit 2 -id 0,1 --
wd 1e-4 --lr 4e-2 --N_grid False --init result/cifar100/res110_32bit/model_best.pth.tar

#2-bit Our grid Training with cifar100
python main.py --arch res110 --dataset cifar100 --w_bit 2 --a_bit 2 -id 0,1 --
wd 1e-4 --lr 4e-2 --N_grid True --init result/cifar100/res110_32bit/model_best.pth.tar

#If you should use cifar10 dataset, you can write --dataset cifar10 