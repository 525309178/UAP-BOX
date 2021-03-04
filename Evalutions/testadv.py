#!/home/administrator/PycharmProjects/venv/bin/python

'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch as t
from torchvision.datasets import ImageFolder
import torchvision.datasets as dset


import argparse
import sys
sys.path.append('../')
from Utils.utils import *
from Classfiers.models import *


parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=32, help='图片的形状')
parser.add_argument('--datatype', type=str, default='adv', help='选择在对抗样本上测试还是原始图像上测试')
parser.add_argument('--targeted', type=int, default=11, help='选择测试准确率还是预测为某个目标标签的准确率')

opt = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best testDemo accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
classfier_path = os.path.abspath(os.path.join(os.getcwd(), "..","Classfiers","CIFAR10","GoogLeNet_cifar10.t7"))


print('==> Preparing data..')
rawdataroot = os.path.abspath(os.path.join(os.getcwd(), "..","RawDatasets"))
advdataroot = os.path.abspath(os.path.join(os.getcwd(), "..","AdversarialExamples","CIFAR10","advganjpg"))


#1 加载对抗样本
transform_test = transforms.Compose([
    transforms.Resize((opt.imageSize, opt.imageSize)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
advset = ImageFolder(advdataroot, transform=transform_test)
advloader = t.utils.data.DataLoader(
                    advset,
                    batch_size=100,
                    shuffle=False,
                    num_workers=2)


#2 加载原始样本
transform_test = transforms.Compose([
    transforms.Resize((opt.imageSize, opt.imageSize)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
rawset = dset.CIFAR10(root=rawdataroot, train=False, download=True, transform=transform_test)
rawloader = torch.utils.data.DataLoader(rawset, batch_size=100, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#3 定义分类器模型并加载已有模型参数
print('==> Building model..')

# net = densenet_cifar()
# net = ResNet50()
# net = VGG('VGG19')
# net = ResNet101()
# net = DenseNet121()
# net = ResNet18()
# net = PreActResNet18()
net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)

if os.path.isfile(classfier_path):
    checkpoint = torch.load(classfier_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    print('load exiting model, the accurate of model is:{:.2f}%'.format(best_acc))
    start_epoch = checkpoint['epoch']
else:
    print('please first train classfier')

# 4 定义损失函数
criterion = nn.CrossEntropyLoss()

if device == 'cuda':
    net.cuda()
    criterion = criterion.cuda()

# 5 模型测试函数
def test(testloader):

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
            if(opt.targeted != 11):
                targets.fill_(opt.targeted)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if(batch_idx % 10 == 0):
                print("===> Accurate[{:.2f}%]    Total: {:.0f}".format(100.*correct/total,total))
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))




if __name__ == '__main__':
    start = time.time()
    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("==> Testing %s.." %(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))))
    # 根据参数判断是在原始图像上测试还是对抗样本上测试
    if opt.datatype == 'adv':
        test(advloader)
    else:
        test(rawloader)
    end = time.time()
    print("==> Test end %s.." %(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))))
