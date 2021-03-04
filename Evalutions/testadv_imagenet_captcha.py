#!/home/user/PycharmProjects/venv/bin/python

import argparse
import random

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

from torch.utils.data import DataLoader




import sys
sys.path.append('../')
from Utils.utils import *
from Classfiers.models import *

parser = argparse.ArgumentParser()
# the param of train process
parser.add_argument('--manualSeed', type=int, default=5198, help='random seed')
parser.add_argument('--workers', type=int, help='workers', default=2)
parser.add_argument('--testBatchSize', type=int, default=2, help='test batch size')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--imageSize', type=int, default=299, help='the weight height of image ')
parser.add_argument('--datatype', type=str, default='adv', help='选择在对抗样本上测试还是原始图像上测试')
parser.add_argument('--targeted', type=int, default=11111, help='选择测试准确率还是预测为某个目标标签的准确率')
parser.add_argument('--targetClassifier', type=str, default='densenet121', help=" classifier (incv3 or vgg16 or vgg19)")
parser.add_argument('--imagenetVal', type=str, default='C:\\Users\\user\\Desktop\\UAP_TOOL\\Attacks\\LPGD_densenet_0.06', help='the path of dataset')
parser.add_argument('--rawimagenetVal', type=str, default='E:\\dataset\\ImageNet\\val\\val_sub', help='the path of dataset')



opt = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best testDemo accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# define normalization means and stddevs
model_dimension = 299 if opt.targetClassifier == 'incv3' else 256
center_crop = 299 if opt.targetClassifier == 'incv3' else 224

mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]



if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

nc = 3

print('==> Preparing data..')
normalize = transforms.Normalize(mean=mean_arr,
                                 std=stddev_arr)

data_transform = transforms.Compose([
    transforms.Resize(model_dimension),
    transforms.CenterCrop(center_crop),
    transforms.ToTensor(),
    normalize,
])

#1 加载对抗样本
advset = torchvision.datasets.ImageFolder(root=opt.imagenetVal, transform=data_transform)
advloader = DataLoader(dataset=advset, num_workers=opt.workers, batch_size=opt.testBatchSize, shuffle=True)

#2 加载原始样本
rawset = torchvision.datasets.ImageFolder(root=opt.rawimagenetVal, transform=data_transform)
rawloader = DataLoader(dataset=rawset, num_workers=opt.workers, batch_size=opt.testBatchSize, shuffle=True)


#3 定义分类器模型并加载已有模型参数
print('==> Building model..')
if opt.targetClassifier == 'incv3':
    net = torchvision.models.inception_v3(pretrained=True)
elif opt.targetClassifier == 'vgg16':
    net = torchvision.models.vgg16(pretrained=True)
elif opt.targetClassifier == 'vgg19':
    net = torchvision.models.vgg19(pretrained=True)
elif opt.targetClassifier == "wide_resnet50_2":
    net = torchvision.models.wide_resnet50_2(pretrained=True)
elif opt.targetClassifier == "googlenet":
    net = torchvision.models.googlenet(pretrained=True)
elif opt.targetClassifier == "densenet121":
    net = torchvision.models.densenet121(pretrained=True)
net.volatile = True

if device == 'cuda':
    net.cuda()

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
    batch_number = 0;
    accattack = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if device == 'cuda':
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if(opt.targeted != 11111):
                targets.fill_(opt.targeted)
            test_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            correct_batch = predicted.eq(targets).sum().item()
            # TODO 计算方法有待核验！！如果该batchsize未全部正确识别，则表示验证码此次未被破解
            if(correct_batch != targets.size(0)):
                accattack += 1
            batch_number += 1

            if(batch_idx % 10 == 0):
                print("===> Accurate[{:.3f}]    Total: {:.0f}".format(accattack/batch_number,batch_number))
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

