from __future__ import print_function
import argparse
import time

import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from models import *
from tqdm import tqdm
from models.slimmableops import bn_calibration_init
import USconfig as FLAGS
import random

#
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default:cifar10 cifar100、minist)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=144, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

#################################### alert this two  path ######################################
parser.add_argument('--resume', default='checkpoints/MobileNetV2/baseline/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--filePath', default='baseline', help='model saved path')
##########################################################################



parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='checkpoints', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='MobileNetV2', type=str,choices=['MobileNet','MobileNetV2','MobileNetV3','USMobileNetV2', 'VGG',
                                                                         'ShuffleNetV2','resnet50'],
                    help='architecture to use')

parser.add_argument('--sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--test',action='store_true')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 赋值后，保存到sr
args.sr = 'True'


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

savepath = os.path.join(args.save, args.arch, args.filePath)
if not os.path.exists(savepath):
    os.makedirs(savepath)
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                          ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data.minist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data.minist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


# 设置CPU
device = torch.device('cpu')
# # 模型和输入数据都需要to device
model = eval(args.arch)(input_size=32)
model = model.to(device)
print(model)

if args.cuda:
    model.cuda()
best_prec1 = -1
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.Adam(model.parameters(), lr=args.lr)
Step_LR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=66, gamma=0.1)

# lr_current = optimizer.param_groups[0]['lr']
# print(lr_current)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        # 修改保存的数据中的学习率为新设定的lr
        # checkpoint['optimizer']['param_groups'][0]['lr'] = 0.01
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f} lr: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1, checkpoint['optimizer']['param_groups'][0]['lr']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))




# with torch.no_grad():
#     for layer in model.modules():
#         if (isinstance(layer, nn.Conv2d)):
#             print(layer.weight)
#
# print("=====")

def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))  # L1


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp




def train():
    model.train()
    avg_loss = 0.
    train_acc = 0.
    test_time = 0.
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        else:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # curr_time1 = time.time()
        output = model(data)
        # curr_time2 = time.time()
        # test_time += curr_time2 - curr_time1
        # print(test_time/(batch_idx+1))

        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        # if args.sr:
        #     updateBN()
        optimizer.step()


def trainUS():
    max_width = max(FLAGS.width_mult_list)
    min_width = min(FLAGS.width_mult_list)

    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        ###
        widths_train = []
        for _ in range(getattr(FLAGS, 'num_sample_training', 2) - 2):
            widths_train.append(
                random.uniform(min_width, max_width))
        widths_train = [max_width, min_width] + widths_train
        # widths_train = [min_width]
        for width_mult in widths_train:
            # TODO :add inplace distillation
            model.apply(lambda m: setattr(
                m, 'width_mult',
                width_mult))
                # always track largest model and smallest model
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
        ###
        optimizer.step()

def test(epoch,test_width=1.0,recal=False):
    model.eval()
    test_loss = 0
    correct = 0
    model.apply(lambda m: setattr(m, 'width_mult',test_width))
    if recal:
        model.apply(bn_calibration_init)
        model.train()
        for idx,(data, target) in enumerate(tqdm(train_loader, total=len(train_loader))):
            if idx==FLAGS.recal_batch:
                break
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output,  = model(data)
                print('===del output==========')
            del output
    model.eval()
    for data, target in tqdm(test_loader, total=len(test_loader)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        else:
            data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nEpoch: {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct.item() / float(len(test_loader.dataset))

def export2normal():
    newmodel=MobileNetV2()
    from collections import OrderedDict
    statedic=[]
    for k2,v in model.state_dict().items():
        if 'running' in k2 or 'num_batches_tracked' in k2:
            continue
        statedic.append(v)
    names=[]
    for k1,v1 in newmodel.state_dict().items():
        if 'running' in k1 or 'num_batches_tracked' in k1:
            continue
        names.append(k1)
    newdic=OrderedDict(zip(names,statedic))
    newmodel.load_state_dict(newdic,strict=False)
    torch.save(newmodel.state_dict(),os.path.join(savepath,'trans.pth'))
    print("save transferred ckpt at {}".format(os.path.join(savepath,'trans.pth')))

best_prec1 = 0. if best_prec1 == -1 else best_prec1
# 实现学习率递减
# scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=0)
if args.test:
    if args.arch=='USMobileNetV2':
        export2normal()
        res_acc=[1.0]*len(FLAGS.width_mult_list)
        for idx,width in enumerate(FLAGS.width_mult_list):
            acc=test(width,recal=True)
            res_acc[idx]=acc
            print("Test accuracy for width {} is {}".format(width,acc))
    else:
        print("Test accuracy {}".format(test()))
else:
    for epoch in range(args.start_epoch, args.epochs):
        if args.arch=='USMobileNetV2':
            trainUS()
            prec1=test(test_width=1.0,recal=False,epoch=epoch)
        else:
            train()
            prec1 = test(epoch=epoch)
            Step_LR.step()
        # scheduler.step(epoch)
        lr_current = optimizer.param_groups[0]['lr']
        print("currnt lr:{}".format(lr_current))
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            ckptfile = os.path.join(savepath, 'model_best.pth.tar')
        else:
            ckptfile = os.path.join(savepath, 'checkpoint.pth.tar')
        # print(model.state_dict())
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, ckptfile)
        torch.save(model, os.path.join(savepath, 'model-cpu.pt'))


# 保存训练的模型可用于移动端
inputTensor = torch.randn(1, 3, 32, 32)
mobileModel = Net(model)
oup = mobileModel(inputTensor)
print(mobileModel)


from thop import clever_format, profile
input = torch.randn(1, 3, 32, 32)
flops, params = profile(model, inputs=(input,), verbose=False)
flops, params = clever_format([flops, params], "%.3f")

print("Total before trainable flops:{}----- params: {}: ", flops, params)

flopsnew, paramsnew = profile(mobileModel, inputs=(input,), verbose=False)
flopsnew, paramsnew = clever_format([flopsnew, paramsnew], "%.3f")

print("main before&&after-delete-Last-Layer flops:{}->{}, params: {}->{}".format(flops, flopsnew, params, paramsnew))

mobileModel.eval()#切换到eval（）
traced_script_module = torch.jit.trace(mobileModel, inputTensor)
traced_script_module.save(os.path.join(savepath, 'model-script.pt'))



# print('测试转换后的图片特征提取===============================================')
# import torch
# import numpy as np
# import math
# # import torchvision.models as models
# from PIL import Image
# image = Image.open("/home/zxq/code/firstTestImage/test_0_52.jpg") #图片发在了build文件夹下
# image = image.resize((32, 32),Image.ANTIALIAS)
# image = np.asarray(image)
# image = image / 255
# image = torch.Tensor(image).unsqueeze_(dim=0)
# image = image.permute((0, 3, 1, 2)).float()
#
# # #------然后把model.pt转换成Pytorch-script，以便在安卓上运行
# # model = torch.load("model-cpu.pt")
# # model.eval()
# input_tensor = torch.rand(1, 3, 32, 32)  # 这里写你的模型的输入张量形状
# script_model = torch.jit.trace(model, input_tensor)
# print('output===================================')