from models.quant_layer_s import *
from models.resnet import *

import numpy as  np
import torch
import sys
from tqdm import tqdm

__all__ = ['update']

def update(model, trainloader, criterion, args):

    if args.QActFlag:
        scaleA = []
    if args.QWeightFlag:
        scaleW = []

    for m in model.modules():
        if isinstance(m, QuantConv2d):
            m.hook = True
            if args.QActFlag:
                scaleA.append(0)

            if args.QWeightFlag:
                scaleW.append(0)
            ### 여기까지 완료
    #print(scaleA)
    #print(scaleW)

    model.train()
    with tqdm(total = 3) as pbar:
        for batch, (images, labels) in enumerate(trainloader):
            if batch == 3:
                break
            images = images.cuda()
            labels = labels.cuda()

            model.zero_grad()
            pred = model(images)
            loss = criterion(pred, labels)
            loss.backward(create_graph=True)

            if args.QActFlag:
                Qact = []
            if args.QWeightFlag:
                Qweight = []

            for m in model.modules():
                if isinstance(m, QuantConv2d):
                    if args.QWeightFlag:
                        Qweight.append(m.buff_weight)
                    if args.QActFlag:
                        Qact.append(m.buff_act)

            if args.QActFlag:
            #Activation
                act_params = []
                act_grad = []
                for i in range(len(Qact)):
                    act_params.append(Qact[i])
                    act_grad.append(Qact[i].grad)

                for i in range(len(Qact)):
                    trace_A = np.mean(trace(model, [act_params[i]], [act_grad[i]])) #이거 받음

                    avg_trace_A = trace_A / act_params[i].view(-1).size()[0]
                    scaleA[i] += (avg_trace_A / (act_grad[i].std().cpu().item()*3.0))

            if args.QWeightFlag:
                weight_params = []
                weight_grad = []
                for i in range(len(Qweight)):
                    weight_params.append(Qweight[i])
                    weight_grad.append(Qweight[i].grad)

                for i in range(len(Qweight)):
                    trace_W = np.mean(trace(model, [weight_params[i]], [weight_grad[i]]))
                    avg_trace_W = trace_W / weight_params[i].view(-1).size()[0]
                    scaleW[i] += (avg_trace_W / (weight_grad[i].std().cpu().item()*3.0))
                pbar.update(1)

        if args.QActFlag:
            for i in range(len(scaleA)):
                scaleA[i] /= batch
                scaleA[i] = np.clip(scaleA[i], 0, np.inf)
            print("\n\nScaleA\n", scaleA)

        if args.QWeightFlag:
            for i in range(len(scaleW)):
                scaleW[i] /= batch
                scaleW[i] = np.clip(scaleW[i], 0, np.inf)
            print("scaleW\n", scaleW)
        print("")

    i = 0
    for m in model.modules():
        if isinstance(m, QuantConv2d):
            if args.QWeightFlag:
                m.factorW.fill_(scaleW[i])
            if args.QActFlag:
                m.factorA.fill_(scaleA[i])
            m.hook = False
            i += 1

def group_product(xs, ys):

    return sum([torch.sum(x * y) for (x,y) in zip(xs, ys)])

def EWGS(grads, param, v):

    hv = torch.autograd.grad(grads, param, grad_outputs=v, only_inputs=True, retain_graph=True)

    return hv

def trace(model, param, grad, maxiter=50, tol=1e-3):
    trace_a = []
    trace = 0.

    for i in range(maxiter):
        model.zero_grad()
        v = [
        torch.randint_like(p, high=2, device='cuda')
        for p in param
        ]

        for v_i in v:
            v_i[v_i == 0] = -1

        H = EWGS(grad, param, v)
        trace_a.append(group_product(H, v).cpu().item())
        if abs(np.mean(trace_a) - trace) / (trace + 1e-6) < tol:
            return trace_a
        else:
            trace = np.mean(trace_a)

    return trace_a
