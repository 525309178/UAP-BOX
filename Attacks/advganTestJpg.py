#!/home/user/PycharmProject/env3.5/bin/python

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image




import time
import sys
sys.path.append('../')
from Utils.utils import *
from Classfiers.models import *
from Attacks.Generator.dcgan import weights_init
from Attacks.Generator.dcgan import Generator
from Attacks.Generator.dcgan import Discriminator

parser = argparse.ArgumentParser()
# the param of train process
parser.add_argument('--manualSeed', type=int, default=999, help='manual seed')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size default=128')
parser.add_argument('--testBatchSize', type=int, default=32, help='testing batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for default=20')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--l2reg', type=float, default=0.01, help='weight factor for l2 regularization')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--imageSize', type=int, default=32, help='the weight height of image')
# the param of attack
parser.add_argument('--optimize_on_success', type=int, default=0, help="whether to optimize on samples that are already successful")
parser.add_argument('--targeted', type=int, default=0, help='targeted attack or not ')
parser.add_argument('--chosen_target_class', type=int, default=1, help='the class to target')
parser.add_argument('--restrict_to_correct_preds', type=int, default=1,
                    help='only compute adv examples on correct predictions')
parser.add_argument('--shrink', type=float, default=0.195, help='scale perturbation')
parser.add_argument('--shrink_inc', type=float, default=0.01,
                    help='update the scale')
parser.add_argument('--max_norm', type=float, default=0.04, help='max allowed perturbation')
parser.add_argument('--norm', type=str, default='linf', help='l2 or linf')
parser.add_argument('--perturbation_type', type=str, default='universal', help='universal" or "imdep" (image dependent)')
parser.add_argument('--adv_weight', type=float, default=1, help='weight of loss')
parser.add_argument('--ldist_weight', type=float, default=0.05, help='weight of loss')
parser.add_argument('--errG_D_weight', type=float, default=0.1, help='weight of loss')
parser.add_argument('--targetClassifier', type=str, default='denseNet', help=" classifier (denseNet or vgg19 or resNet101)")
parser.add_argument('--dataset', type=str, default='cifar10', help='the type of dataset')

# the param of test mode
parser.add_argument('--netG', default='./advgan_uDenseNet/netG_3_0.888.t7', help="the path of netG")
parser.add_argument('--netD', default='./advgan_uDenseNet/netD_3_0.888.t7', help="the path of netD")
parser.add_argument('--mode', type=str, default='test', help='running mode')
parser.add_argument('--save_adv', type=int, default=1, help='save advserial example or not')
parser.add_argument('--outf', default='./advgan_uDenseNet', help='the path of netG and noise')

opt = parser.parse_args()
print(opt)
saveroot = os.path.abspath(os.path.join(os.getcwd(), "..","AdversarialExamples","CIFAR10","advganjpg"))

print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
ngpu = int(opt.ngpu)
torch.cuda.set_device(0)
cudnn.benchmark = True
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


global_acc = 0
mean_arr = [0.4914, 0.4822, 0.4465]
stddev_arr = [0.2023, 0.1994, 0.2010]
# the path of train_log file
try:
    os.makedirs(saveroot)
except OSError:
    pass



print('===> loading dataset..')
if opt.dataset == 'cifar10':
    dataroot = os.path.abspath(os.path.join(os.getcwd(), "..","RawDatasets"))
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = dset.CIFAR10(root=dataroot, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    testset = dset.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.testBatchSize, shuffle=False, num_workers=opt.workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

elif opt.dataset == 'ImageNet':
    print("only support the cifar10 dataset")

print("===> loading targetClassifier ")
if opt.dataset == 'cifar10':
    if(opt.targetClassifier == 'denseNet'):
        target_model = 'Densenet121'
        netClassifier = DenseNet121()
    elif(opt.targetClassifier == 'vgg19'):
        target_model = 'VGG19'
        netClassifier = VGG('VGG19')
    elif(opt.targetClassifier == 'resNet101'):
        target_model = 'Rensenet'
        netClassifier = ResNet101()
    else:
        print("make sure the correct of classfier")

    classfier_path = os.path.abspath(os.path.join(os.getcwd(), "..","Classfiers","CIFAR10","Densenet121_cifar10.t7"))
    checkpoint = torch.load(classfier_path)
    netClassifier.load_state_dict(checkpoint['net'])

elif opt.dataset == 'ImageNet':
    print("only support the cifar10 dataset")

if(opt.cuda):
    netClassifier.cuda()

print("===> Initial Generator and Discriminator ")
nc = 3
netG = Generator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
#print(netG)

if opt.netG != '':
    checkpoint = torch.load(opt.netG,map_location='cuda:0')
    netG.load_state_dict(checkpoint['net'])
    global_acc = checkpoint['acc']
    target_model = checkpoint['target_model']
    print("loading netG chechpoint success ", checkpoint['acc'])
else:
    netG.apply(weights_init)


netD = Discriminator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
#print(netD)
if opt.netD != '':
    checkpoint = torch.load(opt.netD,map_location='cuda:0')
    netD.load_state_dict(checkpoint['net'])
    print("loading netD chechpoint success ", checkpoint['acc'])
else:
    netD.apply(weights_init)

# random noise of train and test
nz = 100
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
criterion = nn.BCELoss()


img_list = []
G_losses = []
D_losses = []
G_accuracy = []
D_prediction = [] # the range is 0-1
iters = 0



def test(epoch, shrink,noise):
    """
    Parameters: epoch          -- 训练轮数
                shrink         -- 扰动缩放系数
                noise          -- 输入的随机噪声
    """
    netG.eval()
    netD.eval()
    netClassifier.eval()

    global global_acc
    # cl_loss 存放制作成功样本的损失，L_inf L2 norm_Ratio raw_L2 adv_L2 分别存放绕动的无穷范数，2范数，噪声2范数与原始图像2范数的比值，原始图像的2范数，对抗样本的2范数
    cl_loss, L_inf, L2,  norm_Ratio, adv_L2, raw_L2 = [], [], [], [], [], []
    total_count, success_count, skipped, no_skipped = 0, 0, 0, 0

    with torch.no_grad():
        # For each batch in the train_loader
        for batch_idx, (inputv, cls) in enumerate(test_loader,0):
            #设置batch-size targets 并将原始数据inputv 和 targets 放置到cuda
            batch_size = inputv.size(0)
            targets = torch.LongTensor(batch_size)
            if opt.cuda:
                inputv = inputv.cuda()
                targets = targets.cuda()
                cls = cls.cuda()
            inputv = Variable(inputv)
            targets = Variable(targets)
            # 获取原始数据 inputv 在分类器处的预测值
            prediction = netClassifier(inputv)
            # 若仅在分类器预测正确的原始样本上进行攻击，更新inputv，prediction，cls（分别对应原始样本，原始样本的预测值，原始样本的真实标签）,删除分类错误的样本
            if opt.restrict_to_correct_preds == 1:
                # get indexes where the original predictions are incorrect
                incorrect_idxs = np.array(np.where(prediction.data.max(1)[1].eq(cls).cpu().numpy() == 0))[0].astype(int)
                skipped += incorrect_idxs.shape[0]
                no_skipped += (batch_size - incorrect_idxs.shape[0])
                if incorrect_idxs.shape[0] == batch_size:
                    # print("All original predictions were incorrect! Skipping batch!")
                    continue
                elif incorrect_idxs.shape[0] > 0 and incorrect_idxs.shape[0] < batch_size:
                    # get indexes of the correct predictions and filter out the incorrect indexes
                    correct_idxs = np.setdiff1d(np.arange(batch_size), incorrect_idxs)
                    correct_idxs = torch.LongTensor(correct_idxs)
                    if opt.cuda:
                        correct_idxs = correct_idxs.cuda()
                    inputv = torch.index_select(inputv, 0, Variable(correct_idxs))
                    prediction = torch.index_select(prediction, 0, Variable(correct_idxs))
                    cls = torch.index_select(cls, 0, correct_idxs)
            # 若是目标攻击则对targets进行赋值，更新inputv，prediction，cls，去除掉与opt.chosen_target_class相同的原数据
            if opt.targeted == 1:
                targets.resize_as_(cls).fill_(opt.chosen_target_class)
                ids = np.array(np.where(targets.data.eq(cls).cpu().numpy() == 0))[0].astype(int)
                ids = torch.LongTensor(ids)
                if opt.cuda:
                    ids = ids.cuda()
                inputv = torch.index_select(inputv, 0, Variable(ids))
                prediction = torch.index_select(prediction, 0, Variable(ids))
                cls = torch.index_select(cls, 0, ids)
            # 更新batch_size，targets，noise的值
            batch_size = inputv.size(0)
            targets.resize_(batch_size)
            noise.resize_(batch_size, nz, 1, 1)

            # 过滤完成,开始制作对抗样本，inputv，prediction，cls，targets，noise均可直接使用
            ############################
            # 将原始数据重新命名为 real_cpu
            real_cpu = inputv.to(device)
            adv_per = netG(noise).to(device)
            fake = torch.add(adv_per * shrink, real_cpu)
            # 对攻击样本进行修剪
            for cii in range(3):
                fake.data[:, cii, :, :] = fake.data[:, cii, :, :].clamp(real_cpu.data[:, cii, :, :].min(),
                                                                        real_cpu.data[:, cii, :, :].max())

            adv_prediction = netClassifier(fake)
            save_adv_sample = fake.clone()
            # ######################  save adv_sample #########################
            if opt.save_adv == 1:
                for i in range(0, batch_size):
                    save_adv = save_adv_sample[i]
                    # undo normalize image color channels
                    for c2 in range(3):
                        save_adv.data[c2, :, :] = (save_adv.data[c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
                    # save_adv = rescale(save_adv, mean, std)
                    label_path = str(int(cls[i]))
                    if not os.path.exists(os.path.join(saveroot, label_path)):
                        os.mkdir(os.path.join(saveroot, label_path))

                    ndarr = save_adv.data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    im = Image.fromarray(ndarr)
                    im.save('{}/{}/{}_{}.jpg'.format(saveroot, label_path, batch_idx, i),quality=10)

                    # vutils.save_image(save_adv.data,
                    #                   '{}/{}/{}_{}.png'.format(saveroot, label_path, batch_idx, i))  # 保存3D 0-1 tensor 为图片格式
            # ######################  End save adv_sample #########################

            # 获取到制作失败对抗样本对应的索引
            if opt.targeted == 1:
                targets.resize_as_(cls).fill_(opt.chosen_target_class)
                no_idx = np.array(np.where(adv_prediction.data.max(1)[1].eq(targets.data).cpu().numpy() == 0))[0].astype(
                    int)
            else:
                no_idx = np.array(np.where(adv_prediction.data.max(1)[1].eq(prediction.data.max(1)[1]).cpu().numpy() == 1))[
                    0].astype(int)

            # 更新制作的总数与制作成功的数量
            success_count += real_cpu.size(0) - len(no_idx)
            total_count += real_cpu.size(0)
            # 计算对抗绕动的范数信息，L_inf L2 norm_Ratio raw_L2 adv_L2 分别存放绕动的无穷范数，2范数，噪声2范数与原始图像2范数的比值，原始图像的2范数，对抗样本的2范数
            if len(no_idx) != real_cpu.size(0):
                yes_idx = np.setdiff1d(np.array(range(real_cpu.size(0))), no_idx)
                for i, adv_idx in enumerate(yes_idx):
                    clean = real_cpu[adv_idx].data.view(1, nc, opt.imageSize, opt.imageSize)
                    adv = fake[adv_idx].data.view(1, nc, opt.imageSize, opt.imageSize)
                    pert = (real_cpu[adv_idx] - fake[adv_idx]).data.view(1, nc, opt.imageSize, opt.imageSize)

                    # 对抗样本与原始样本去标准化使用的为3D-Tensor,用于计算范数
                    rec_adv = rescale(fake[adv_idx], mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                    rec_clean = rescale(real_cpu[adv_idx], mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

                    # 绕动的无穷范数（原始样本与对抗样本之间差距最大值）
                    linf = torch.max(torch.abs(rec_adv - rec_clean)).data.cpu().numpy()
                    # 绕动的2范数（原始样本与对抗样本之间的欧式距离）
                    noise_norm = torch.sqrt(torch.sum((rec_clean[:, :, :] - rec_adv[:, :, :]) ** 2)).data.cpu().numpy()
                    # 原始样本的2范数
                    image_norm = torch.sqrt(torch.sum(rec_clean[:, :, :] ** 2)).data.cpu().numpy()
                    # 对抗样本的2范数
                    adv_norm_s = torch.sqrt(torch.sum(rec_adv[:, :, :] ** 2)).data.cpu().numpy()

                    # 噪声2范数与原始样本2范数的比值
                    norm_Ratio.append(noise_norm / image_norm)
                    # 噪声2范数的值
                    L2.append(noise_norm)
                    # 原始样本的2范数
                    raw_L2.append(image_norm)
                    # 对抗样本的2范数
                    adv_L2.append(adv_norm_s)
                    # 噪声的无穷范数(目前使用到的度量）
                    L_inf.append(linf)
                    if i == 0:
                        vutils.save_image(torch.cat((clean, pert, adv)), './{}/{}_{}.png'.format(opt.outf, epoch, i),
                                          normalize=True, scale_each=True)
            # train_log to file
            progress_bar(batch_idx, len(test_loader), "Val E%s, C_L %.2f A_Succ %.5f L_inf %.5f L2 %.5f (Ratio %.2f, AdvL2 %.2f, CleanL2 %.2f) C %.3f Skipped %.2f%%"
                         %(epoch, np.mean(cl_loss), success_count/total_count, np.mean(L_inf), np.mean(L2), np.mean(norm_Ratio), np.mean(adv_L2), np.mean(raw_L2), shrink, 100*(skipped/(skipped+no_skipped))))



if __name__ == '__main__':

    c = opt.shrink
    prev_pred = 0
    if(opt.mode == 'train'):
        print("please set the mode is test",target_model)

    else:
        start = time.time()
        test(1, c ,fixed_noise)
        end = time.time()
