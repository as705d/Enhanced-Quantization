# Enhanced-Quantization
* We propose a new quantization technique inspired by APOT ([Additive Powers-of-Two](https://arxiv.org/pdf/1909.13144.pdf)).  
* We can use low-bit (e.g. 2-bit) to powers-of-two and improve accuracy at the same time.  
* We also approach the problem of weight.

<p align="center">
<img src="https://user-images.githubusercontent.com/51831143/185300574-94f63f11-891d-4d22-9036-bb2fae4311f0.png">
</p>

### Our method
```python
import torch
import numpy as np

def weight_grid_setting(B=2, N_grid=False, Z=2):
  weight_grids = []
  
  if B = 1:
    if N_grid:
      weight_grids.append(1)
      weight_grids.append(2 ** (-B * Z))
    else:
      weight_grids.append(0)
      weight_grids.append(1)
      
  else:
    NotImplementedError
```
Our idea is defined in the ```weight_grid_setting``` function.  
It also applies only to 2-bit weight.   
If the ```N_grid = True```, our idea applies. In the opposite case, the existing method applies.  
```Z > 0``` and you are free to choose.

# Usage Environment
* python 3.8.12
* pytorch 1.8.0
* CUDA 11.1

# DataSet
* CIFAR10 and CIFAR100
* CIFAR10 and 100 are automatically downloaded at run.


# Run
* 32-bit run with CIFAR100
```python 
python main.py --arch res56 --dataset cifar100 --w_bit 32 -id 0,1 --wd 1e-4
```

* 2-bit run with CIFAR100 by our method.
```python
python main.py --arch res56 --dataset cifar100 --w_bit 2 --a_bit 2 -id 0,1 --
wd 1e-4 --lr 4e-2 --N_grid True --constant 2 --init result/cifar100/res56_32bit/model_best.pth.tar
```

* To evaluate  
The constant values used for training and testing must be the same.
```python
python main.py -e --arch res56 --init result/cifar100/res56_2bit/N_grid/model_best.pth.tar -e -id 0 --w_bit 2 --a_bit 2 --
N_grid True --constant 2 --dataset cifar100
```

# Result
* ResNet20  
| Model | W2A2 | Z | Acc |
| ----- | --- | --- | --- |
| ResNet20 | W2A2 | 1 | 90.45 |
| ResNet20 | W2A2 | 2 | 90.80 |
| ResNet20 | W2A2 | 4 | 90.50 |
| ResNet20 | W2A2 | 10 | 90.45 |
| ResNet20 | W2A2 | 20 | 89.59 |

* ResNet32  
Model | W2A2 | Z | Acc
ResNet32 | --- | 1 | 67.25
ResNet32 | --- | 2 | 68.06
ResNet32 | --- | 4 | 67.85
ResNet32 | --- | 10 | 67.32
ResNet32 | --- | 20 | 67.24

* ResNet56  
Model | W2A2 | Z | Acc
ResNet56 | --- | 1 | 69.64
ResNet56 | --- | 2 | 69.80
ResNet56 | --- | 4 | 69.69
ResNet56 | --- | 10 | 69.51
ResNet56 | --- | 20 | 68.81

|제목|내용|설명|
|------|---|---|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
