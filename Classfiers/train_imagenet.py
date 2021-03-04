#!/home/user/PycharmProjects/venv/bin/python

import argparse
import random

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim


from torch.utils.data import DataLoader




import sys
sys.path.append('../')
from Utils.utils import *
from Classfiers.models import *

parser = argparse.ArgumentParser()
# the param of train process
parser.add_argument('--manualSeed', type=int, default=5198, help='random seed')
parser.add_argument('--workers', type=int, help='workers', default=2)
parser.add_argument('--testBatchSize', type=int, default=32, help='test batch size')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--imageSize', type=int, default=299, help='the weight height of image ')
parser.add_argument('--datatype', type=str, default='adv', help='选择在对抗样本上测试还是原始图像上测试')
parser.add_argument('--targeted', type=int, default=11111, help='选择测试准确率还是预测为某个目标标签的准确率')
parser.add_argument('--targetClassifier', type=str, default='densenet121', help=" classifier (incv3 or vgg16 or vgg19)")
parser.add_argument('--imagenetVal', type=str, default='C:\\Users\\user\\Desktop\\UAP_TOOL\\Attacks\\FGSM_DenseNet121_0.05', help='the path of dataset')
parser.add_argument('--rawimagenetVal', type=str, default='E:\\dataset\\ImageNet\\val\\val_sub', help='the path of dataset')
parser.add_argument('--model_path', default='./retrainModel/DenseNet_Imagenet.t7', help='path for save model or load model')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')




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

optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

if os.path.isfile(opt.model_path):
    print('load exiting model')
    checkpoint = torch.load(opt.model_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Training
def train(epoch):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(advloader):
        # 控制可获取到的对抗验证码数量，2000,4000,6000,8000,10000
        if batch_idx > 62:
            break
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

        progress_bar(batch_idx, len(advloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch):

    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(advloader):
            if device == 'cuda':
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(advloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
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
        if not os.path.isdir('retrainModel'):
            os.mkdir('retrainModel')
        torch.save(state, '%s/_%.3f.t7' % (opt.model_path,acc))
        best_acc = acc

 #  --------------------------------------------  Training --------------------------------------------
if __name__ == '__main__':

	for epoch in range(start_epoch, start_epoch+6):
		start = time.time()
		time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
		print("==> Training %s.." %(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))))
		train(epoch)
		test(epoch)
		end = time.time()
		print("==> Epoch end time %s.." %(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))))