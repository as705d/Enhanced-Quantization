# Enhanced-Quantization

# Usage Environment
python 3.8.12
pytorch 1.8.0
CUDA 11.1

# DataSet
Cifar10 and Cifar100

32bit run with Cifar100
```python 
python main.py --arch res20 --dataset cifar10 --epochs 200 --w_bit 32 -id 0,1 --wd 1e-4
```


