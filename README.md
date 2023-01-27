# Enhanced-Quantization
* We propose a new quantization technique inspired by APOT ([Additive Powers-of-Two](https://arxiv.org/pdf/1909.13144.pdf)).  
* We can use low-bit (e.g. 2-bit) to powers-of-two and improve accuracy at the same time.  
* We also approach the problem of weight values.

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

* ResNet56 with CIFAR100 (32-bit: **71.77%**)

|**Model**|**W2A2**|**Z**|**Acc(%)**|
|:------:|:---:|:---:|:---:|
|ResNet56|---|1|69.64(%)|
|ResNet56|---|2|**69.87(%)**|
|ResNet56|---|4|69.75(%)|
|ResNet56|---|10|69.51(%)|
|ResNet56|---|20|68.81(%)|

* **Weight Clipping (Existing)**  
```
wgt_alpha: [0.858, -0.418, 0.848, -0.764, 0.783, -0.383, 0.728, 0.738, 0.618, 0.707, 0.771, -0.643, 0.629, 0.67, 0.581, 0.676, 0.
583, 0.601, 0.596, 0.651, 0.708, 0.797, 0.89, 0.818, 0.873, 0.845, 0.887, 0.79, 0.936, 0.773, 0.927, 0.782, 0.891, 0.807, 0.842,
0.796, 0.935, 0.707, 0.7, -0.632, 0.873, 0.823, 0.856, 0.86, 0.85, 0.835, 0.838, 0.873, 0.843, 0.853, 0.828, 0.87, 0.811, 0.87, 0
.791, 0.799]
```

* **Weight Clipping (Our)**  
```
wgt_alpha: [1.003, 1.009, 0.909, 0.864, 0.958, 1.02, 0.908, 0.985, 0.932, 1.006, 0.802, 0.956, 0.895, 0.929, 0.807, 0.905, 0.777, 0.
871, 0.793, 0.861, 0.383, 1.032, 1.126, 0.998, 1.114, 1.073, 1.111, 1.042, 1.098, 0.966, 1.127, 0.972, 1.136, 1.013, 1.095, 1.006, 1.118,
0.938, 0.935, 0.506, 1.082, 1.02, 1.077, 1.102, 1.042, 1.041, 1.053, 1.056, 1.04, 1.073, 1.026, 1.085, 1.003, 1.079,0.998, 1.01]
```
