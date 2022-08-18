import argparse
import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

import torchvision
import torchvision.transforms as transforms

#from ptflops import get_model_complexity_info
from models import *
from update import *
#import visdom

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10','cifar100'), help='dataset to use CIFAR10|CIFAR100')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='res20')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('--init', help='initialize form pre-trained floating point model', type=str, default='')
parser.add_argument('-id', '--device', default='0', type=str, help='gpu device')
parser.add_argument('--w_bit', default=32, type=int, help='the bit-width of the quantized network')
parser.add_argument('--a_bit', default=32, type=int, help='the bit-width of the quantized network')
parser.add_argument('--constant', default=2, type=int, help='exponential product')
parser.add_argument('--factorW', type=float, default=0.0, help='scaling factor for weights')
parser.add_argument('--factorA', type=float, default=0.0, help='scaling factor for activations')
parser.add_argument('--quantize', type=str2bool, default=False, help='training with STE')
parser.add_argument('--EWGS', type=str2bool, default=False, help='Apply with EWGS') #We don't use EWGS 
parser.add_argument('--N_grid', type=str2bool, default=False, help='Apply weight-bit with Our grid')
#parser.add_argument('--update_every', type=int, default=10, help='update interval in terms of epochs')
parser.add_argument('--QWeightFlag', type=str2bool, default=True, help='do weight quantization')
parser.add_argument('--QActFlag', type=str2bool, default=True, help='do activation quantization')

best_prec = 0
best_prec5 = 0
args = parser.parse_args()

def main():

    global args, best_prec, best_prec5
    use_gpu = torch.cuda.is_available()
        
    if args.dataset == 'cifar10':
        args.num_classes = 10
        print('=> loading cifar10 data...')
            
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        
    elif args.dataset == 'cifar100':
        print('=> loading cifar100 data...')
        args.num_classes = 100
        if not os.path.exists('result/'+str(args.dataset)):
            os.makedirs('result/'+str(args.dataset))
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        raise NotImplementedError
        
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    print("=> data loading finished...")
    
    print(args.device)
    print('=> Building model...')
    model=None
    if use_gpu:
        float = True if args.w_bit == 32 else False
        if args.arch == 'res20':
            model = resnet20_cifar(args, float=float)         

        elif args.arch == 'res56':
            model = resnet56_cifar(args, float=float)

        elif args.arch == 'res32':
            model = resnet32_cifar(args, float=float)
            
        elif args.arch == 'res110':
            model = resnet110_cifar(args, float=float)
                        
        else:
            print('Architecture not support!')
            return
        if not float:
            for m in model.modules():
                if isinstance(m, QuantConv2d):
                    in_channels = m.in_channels
                    out_channels = m.out_channels
                    m.weight_quant = weight_quantize_fn(factorW=args.factorW, w_bit=args.w_bit, N_grid=args.N_grid,
                                                        constant=args.constant)
                    m.act_grid = act_grid_setting(args.a_bit)
                    m.act_alq = act_quantization(args.factorA, args.a_bit, m.act_grid)

        model = nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        model_params = []
        for name, params in model.module.named_parameters():
            if 'act_alpha' in name:
                model_params += [{'params': [params], 'lr': 1e-1, 'weight_decay': 1e-4}]
            elif 'wgt_alpha' in name:
                model_params += [{'params': [params], 'lr': 2e-2, 'weight_decay': 1e-4}]
            else:
                model_params += [{'params': [params]}]
        optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    if not os.path.exists('result/'+str(args.dataset)):
        os.makedirs('result/'+str(args.dataset))  
        
    if args.dataset == 'cifar10':
        if args.N_grid:
            fdir = 'result/'+str(args.dataset)+'/'+str(args.arch)+'_'+str(args.w_bit)+'bit'+'/N_grid'
            if not os.path.exists(fdir):
                os.makedirs(fdir)
        else:
            fdir = 'result/'+str(args.dataset)+'/'+str(args.arch)+'_'+str(args.w_bit)+'bit'
            if not os.path.exists(fdir):
                os.makedirs(fdir)
                
    elif args.dataset == 'cifar100':
        if args.N_grid:
            fdir = 'result/'+str(args.dataset)+'/'+str(args.arch)+'_'+str(args.w_bit)+'bit'+'/N_grid'
            if not os.path.exists(fdir):
                os.makedirs(fdir)
        else:
            fdir = 'result/'+str(args.dataset)+'/'+str(args.arch)+'_'+str(args.w_bit)+'bit'
            if not os.path.exists(fdir):
                os.makedirs(fdir)
    else:
        raise NotImplementedError
        
    if args.init:
        if os.path.isfile(args.init):
            print("=> loading pre-trained model")
            checkpoint = torch.load(args.init)
            model.load_state_dict(checkpoint['state_dict'],strict=False)
        else:
            print('No pre-trained model found !')
            exit()

    if args.evaluate:
        validate(testloader, model, criterion)
        alpha_list = model.module.show_params()
        return
    
    writer = SummaryWriter(comment=fdir.replace('result/', ''))
    iter = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        #if epoch % args.update_every == 0 and epoch != 0 and args.EWGS:
        #    update(model, trainloader, criterion, args)
        # train for one epoch
        #model.module.record_weight(writer, epoch)
        if epoch%10 == 1:
            model.module.show_params()

        # model.module.record_clip(writer, epoch)
        iteration = train(trainloader, model, criterion, optimizer, epoch, args.device, args, writer, iter)
        # evaluate on test set
        prec, prec5 = validate(testloader, model, criterion)
        writer.add_scalar('test_acc', prec, epoch)

        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec,best_prec)

        top3_best = prec5 > best_prec5
        best_prec5 = max(prec5, best_prec5)
        print('best acc: {:1f} \t best acc5: {:1f}'.format(best_prec, best_prec5))
        
        #Save Model
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec': best_prec,
                'optimizer': optimizer.state_dict(),
            }, is_best, fdir, epoch)

        iter = iteration

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(trainloader, model, criterion, optimizer, epoch, device, args, writer, iter):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.QWeightFlag:
        scaleW = []
    end = time.time()
    for m in model.modules():
        if isinstance(m, QuantConv2d):
            m.hook = True
            if args.QWeightFlag:
                scaleW.append(0)
                
    model.train()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        #output : [0, 99]
        
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()                                               
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        writer.add_scalar('Train_loss',loss.item() ,iter)
        #weight gradient output
        
        iter += 1
        # if i % 2 == 0:
        #     model.module.show_params()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                  'Prec5 {top5.val:.3f}% ({top5.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            
            
    return iter

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for m in model.modules():
        if isinstance(m, QuantConv2d):
            m.hook = False
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec, prec5 = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                  'Prec5 {top5.val:.3f}% ({top5.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec {top1.avg:.3f}% \t Prec5 {top5.avg:.3f}%'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, fdir, epoch):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    adjust_list = [150, 225]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True) #top k
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()
