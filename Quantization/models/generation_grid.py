import torch
import numpy as np

def weight_grid_setting(B=2, N_grid=False, Z=2):
    #https://arxiv.org/pdf/1909.13144.pdf in APOT
    #3bit and 4bit are applied the APOT 
    #2bit is applied the new grid
    #grid is generated 2^b - 1
    # Z > 0 
    
    base_a = [0.]
    base_b = [0.]
    weight_grids = []

    if B == 3: #4bit weight (because weight is B-1)
        for i in range(3):
            if i < 2:
                base_a.append(2 ** (-i - 1))
            else:
                base_b.append(2 ** (-i - 1))
                base_a.append(2 ** (-i - 2))
    elif B == 2: #3bit weight
        for i in range(3):
            base_a.append(2 ** (-i - 1))

    elif B == 1: #2bit weight --> Apply the new grid
        if N_grid: #Our Grid : [1, 2^(-BxZ)], Therefore, If Z = 2, quantized weight is [-3, -0.75, 0.75, 3]
            #2^(-BxZ) is nearest zero.
            weight_grids.append(1)
            weight_grids.append(2 ** (-B * Z))
        else: #Grid of APOT : [0, 1], Therefore quantized weight is [-3, -0, 0, 3]
            weight_grids.append(0)
            weight_grids.append(1)
        
    #elif B == 0: #1bit weight
    #    weight_grids.append(1)

    else:
        raise NotImplementedError
        
    if B == 3 or B == 2: #existing method.
        for a in base_a:
            for b in base_b:
                weight_grids.append(a + b)
        weight_grids = torch.Tensor(list(set(weight_grids)))
        weight_grids.mul(1.0 / torch.max(weight_grids))
        #weight_grids[0] = 2 ** (-B * z)
        return weight_grids

    else:
        return torch.Tensor(list(set(weight_grids)))


def act_grid_setting(B=2):
    #Activation function is applied only APOT
    base_a = [0.]
    base_b = [0.]
    act_values = []
    if B == 4: #4bit Activation (because Activation is B)
        for i in range(3):
            base_a.append(2 ** (-2 * i - 1))
            base_b.append(2 ** (-2 * i - 2))
    elif B == 3: #3bit Activation
        for i in range(3):
            if i < 2:
                base_a.append(2 ** (-i - 1))
            else:
                base_b.append(2 ** (-i - 1))
                base_a.append(2 ** (-i - 2))
    elif B == 2: #2bit Activation
        for i in range(3):
            base_a.append(2 ** (-i - 1))

    else:
        raise NotImplementedError

    for a in base_a:
        for b in base_b:
            act_values.append(a + b)

    act_values = torch.Tensor(list(set(act_values)))
    act_values = act_values.mul(1.0 / torch.max(act_values))
    return act_values
