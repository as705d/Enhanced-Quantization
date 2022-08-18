import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import time
from models.generation_grid import weight_grid_setting, act_grid_setting

# this function construct an additive pot quantization levels set, with clipping threshold = 1,

def weight_quantization(b, grids, factorW):

    def grid_quant(x, value_s):
    
        #values is grids
        #print(value_s) #[0,1]
        shape = x.shape #layer shape
        xhard = x.view(-1) #xhard is 
        value_s = value_s.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1] # project to nearest quantization level
        xhard = value_s[idxs].view(shape) #Select value in grid 
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            #input is normalization weight and alpha is fixed to 3
            
            input.div_(alpha)                          # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)       # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_q = grid_quant(input_abs, grids).mul(sign)  # project to Q^w(alpha, B)
            
            input_out = input_q.mul(alpha)               # rescale to the original range
            #not_zero = torch.count_nonzero(input_out)
            #total = (input_out).view(-1).size()[0]
            #zero_count = total - not_zero #zero_count
            
            ctx._factorW = factorW
            ctx.save_for_backward(input, input_q)
            return input_out

        @staticmethod
        def backward(ctx, grad_output):
            
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            scale = ctx._factorW
            
            #weight_scale =  1 + scale * torch.sign(grad_input) * (input_q - input) #EWGS
            i = (input.abs()>1.).float()
            sign = input.sign()
            grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            
            return grad_input, grad_alpha

    return _pq().apply

class weight_quantize_fn(nn.Module):
    def __init__(self, factorW, w_bit, N_grid, constant):
        super(weight_quantize_fn, self).__init__()
        assert (w_bit <=5 and w_bit > 0) or w_bit == 32
        self.factorW = factorW
        self.w_bit = w_bit - 1 #b = b-1 (because using sign function)
        self.register_parameter('wgt_alpha', Parameter(torch.tensor(3.0)))
        self.grids = weight_grid_setting(self.w_bit, N_grid, constant)
        self.weight_q = weight_quantization(b=self.w_bit, grids=self.grids, factorW=self.factorW)
        

    def forward(self, weight):
        if self.w_bit == 32:
            weight_q = weight
        else:
            mean = weight.data.mean()
            std = weight.data.std()
            weight = weight.add(-mean).div(std)      # weights normalization
            
            weight_q = self.weight_q(weight, self.wgt_alpha) #wgt_alpha is fixed to 3
        return weight_q


def act_quantization(factorA, b, grid):

    def grid_quant(x, grid):
        #grid : 2^b
        shape = x.shape
        xhard = x.view(-1)
        value_s = grid.type_as(x) 
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        xhard = value_s[idxs].view(shape)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            
            input=input.div(alpha)
            input_c = input.clamp(max=1)
            input_q = grid_quant(input_c, grid)

            input_out = input_q.mul(alpha)
            ctx.save_for_backward(input, input_q)
            ctx._factorA = factorA
            
            return input_out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            scale = ctx._factorA
            
            #act_scale = 1 + scale * torch.sign(grad_input) * (input_q - input)
            i = (input > 1.).float()
            grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
            grad_input = grad_input*(1-i)
            
            return grad_input, grad_alpha

    return _uq().apply
        
class QuantConv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, args, stride=1, padding=0, dilation=1, groups=1, bias=False):

        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)
        self.layer_type = 'QuantConv2d'
        self.w_bit = 4
        self.a_bit = 4
        self.register_buffer('factorW', torch.tensor(args.factorW).float())
        self.register_buffer('factorA', torch.tensor(args.factorA).float())
        self.quan_weight = args.QWeightFlag
        self.quan_act = args.QActFlag
        self.N_grid = args.N_grid
        self.constant = args.constant
        
        if self.quan_weight:
            self.weight_quant = weight_quantize_fn(self.factorW, w_bit=self.w_bit, N_grid=self.N_grid,
                                                   constant=self.constant)
            
        if self.quan_act:
            self.act_grid = act_grid_setting(self.a_bit)
            self.act_alq = act_quantization(self.factorA, self.a_bit, self.act_grid)
            self.act_alpha = torch.nn.Parameter(torch.tensor(8.0))
            
        self.hook = False
        self.buff_weight = None
        self.buff_act = None

        
    def forward(self, x):
        
        weight_q = self.weight_quant(self.weight)
        x = self.act_alq(x, self.act_alpha)
        
        if self.hook:
            self.buff_weight = weight_q
            self.buff_weight.retain_grad()
            self.buff_act = x
            self.buff_act.retain_grad()
        
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def show_params(self):
        
        wgt_alpha = round(self.weight_quant.wgt_alpha.data.item(), 3)
        act_alpha = round(self.act_alpha.data.item(), 3)
        return wgt_alpha, act_alpha

# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, args, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FConv2d'
    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q-self.weight).detach()+self.weight
        return F.linear(x, weight_q, self.bias)