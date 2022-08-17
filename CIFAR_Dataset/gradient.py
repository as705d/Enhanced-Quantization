import torch
import numpy as np
import time

def gradient(param, FP32, grad, scaling):
    a = torch.count_nonzero(grad)
    b = (grad).view(-1).size()[0]
    count = b - a

    if grad.dim() != 1 and count > 0: #gradient = 0
        zp = (grad == 0).nonzero(as_tuple=True)
        zero_g = grad[zp] #[0,0,0,0,0,0,0,0]
        param_i = param[zp]
        FP32_i = FP32[zp] 
        
        #print("param:", param[0][0][0][0])
        #print("FP32_i:", FP32[0][0][0][0])
        weight_loss = FP32_i - param_i
        if weight_loss.pow(2).mean() == 0:
            grad = grad
        else:
            sign = torch.sign(FP32_i)
            direction = sign * weight_loss
            zero_g = zero_g + (direction * scaling)
            grad[zp] = zero_g                     
            #print(grad[zp])
    return grad
        
        
