# Enhanced-Quantization
* We propose a new quantization technique inspired by APOT ([Additive Powers-of-Two](https://arxiv.org/pdf/1909.13144.pdf)).  
* We can use low-bit (e.g. 2-bit) to powers-of-two and improve accuracy at the same time.  
* We also approach the problem of weight.

<p align="center">
<img src="https://user-images.githubusercontent.com/51831143/185300574-94f63f11-891d-4d22-9036-bb2fae4311f0.png">
</p>

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
# Usage Environment
* python 3.8.12
* pytorch 1.8.0
* CUDA 11.1

# DataSet
* Cifar10 and Cifar100
* Cifar10 and 100 are automatically downloaded at run.


# Run
* 32bit run with Cifar100
```python 
python main.py --arch res20 --dataset cifar10 --epochs 200 --w_bit 32 -id 0,1 --wd 1e-4
```


