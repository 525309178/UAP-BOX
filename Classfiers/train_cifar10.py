#!/home/administrator/PycharmProjects/venv/bin/python

'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch as t
import torchvision as tv
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms


import argparse
import os
import time
from models import *

import sys
sys.path.append('../')
from Utils.utils import progress_bar


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--model_path', default='./CIFAR10/MobileNet_cifar10.t7', help='path for save model or load model')


opt = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best testDemo accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# 加载数据集
print('==> Preparing data..')
dataroot = os.path.abspath(os.path.join(os.getcwd(), "..","RawDatasets"))
print(dataroot)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = tv.datasets.CIFAR10(
                    root=dataroot,
                    train=True,
                    download=True,
                    transform=transform_train)
trainloader = t.utils.data.DataLoader(
                    trainset,
                    batch_size=64,
                    shuffle=True,
                    num_workers=2)

testset = tv.datasets.CIFAR10(
                    root=dataroot,
                    train=False,
                    download=True,
                    transform=transform_test)

testloader = t.utils.data.DataLoader(
                    testset,
                    batch_size=100,
                    shuffle=False,
                    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ----------------------------------------------ＳＴＥＰ 2：Define classfier model --------------------------------------------
print('==> Building model..')

# net = densenet_cifar()
# net = ResNet50()
# net = VGG('VGG19')
# net = ResNet101()
# net = DenseNet121()

# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
#net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(1)

if os.path.isfile(opt.model_path):
    print('load exiting model')
    checkpoint = torch.load(opt.model_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# ----------------------------------------------ＳＴＥＰ３:Define loss function and optim --------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

# ---------------------------------------------ＳＴＥＰ 4: Define cuda if exit --------------------------------------------
if device == 'cuda':
    net.cuda()
    criterion = criterion.cuda()

# Training
def train(epoch):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if device == 'cuda':
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch):

    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if device == 'cuda':
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('CIFAR10'):
            os.mkdir('CIFAR10')
        torch.save(state, opt.model_path)
        best_acc = acc



 #  --------------------------------------------ＳＴＥＰ5 :Training --------------------------------------------
if __name__ == '__main__':
	for epoch in range(start_epoch, start_epoch+30):

		start = time.time()
		time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
		print("==> Training %s.." %(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))))
		train(epoch)
		test(epoch)
		end = time.time()
		print("==> Epoch end time %s.." %(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))))
