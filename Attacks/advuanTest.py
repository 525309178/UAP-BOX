#!/home/user/PycharmProjects/venv/bin/python

import argparse
import random

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import time
import sys
sys.path.append('../')
from Utils.utils import *
from Classfiers.models import *
from Classfiers.models import LossNetwork
from Attacks.Generator.uanG import _netAttacker

###############################################################################
#  Reference: Learning Universal Adversarial Perturbations with Generative Models
#  This is just a simple  test , please read the original paper for more details
###############################################################################

parser = argparse.ArgumentParser()
# the param of train process
parser.add_argument('--manualSeed', type=int, default=5198, help='random seed')
parser.add_argument('--workers', type=int, help='workers', default=8)
parser.add_argument('--batchSize', type=int, default=32, help='train batch size')
parser.add_argument('--testBatchSize', type=int, default=32, help='test batch size')
parser.add_argument('--epochs', type=int, default=500, help='epoch')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='the param of optim')
parser.add_argument('--l2reg', type=float, default=0.01, help='the param of optim')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--imageSize', type=int, default=32, help='the weight height of image ')
# the param of attack
parser.add_argument('--optimize_on_success', type=int, default=0, help="whether to optimize on samples that are already successful")
parser.add_argument('--targeted', type=int, default=0, help='targeted attack or not')
parser.add_argument('--chosen_target_class', type=int, default=1, help='the class to target')
parser.add_argument('--restrict_to_correct_preds', type=int, default=1,
                    help='only compute adv examples on correct predictions')
parser.add_argument('--shrink', type=float, default=0.009, help='scale perturbation ')
parser.add_argument('--shrink_inc', type=float, default=0.01,
                    help='update the scale')
parser.add_argument('--max_norm', type=float, default=0.04, help='max allowed perturbation')
parser.add_argument('--norm', type=str, default='linf', help='l2 or linf')
parser.add_argument('--perturbation_type', type=str, default='universal', help='"universal" or "imdep" (image dependent)')
parser.add_argument('--nz', type=int, default=100, help='the size of noise')
parser.add_argument('--lfs_weight', type=float, default=1.0, help='weight of loss')
parser.add_argument('--ldist_weight', type=float, default= 2.0, help='weight of loss')
parser.add_argument('--content_weight', type=float, default=1, help='weight of loss')
parser.add_argument('--targetClassifier', type=str, default='denseNet', help=" classifier (denseNet or vgg19 or resNet101)")
parser.add_argument('--dataset', type=str, default='cifar10', help='the type of dataset')

# the param of test mode
parser.add_argument('--netAttacker', default='./advuan_DenseNet/netAttacker_4_0.83484.t7', help="the path of netAttacker")
parser.add_argument('--mode', type=str, default='test', help='running mode')
parser.add_argument('--save_adv', type=int, default=0, help='save advserial example or not')
parser.add_argument('--outf', default='./advuan_DenseNet', help='the path of netAttacker')



opt = parser.parse_args()
print(opt)
torch.cuda.set_device(0)
global_acc = 0
mean_arr = [0.4914, 0.4822, 0.4465]
stddev_arr = [0.2023, 0.1994, 0.2010]


try:
    os.makedirs(opt.outf)
except OSError:
    pass

WriteToFile('./%s/train_log' % (opt.outf), opt)

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


print("==> Initial netAttacker ")
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.1)

netAttacker = _netAttacker(0,opt.imageSize)
netAttacker.apply(weights_init)
if opt.netAttacker != '':
    checkpoint = torch.load(opt.netAttacker,map_location='cuda:0')
    netAttacker.load_state_dict(checkpoint['net'])
    global_acc = checkpoint['acc']
    target_model = checkpoint['target_model']
    print("loading netAttacker chechpoint success ", checkpoint['acc'])

#print(netAttacker)
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

print("===> loading lossnetwork")
if opt.dataset == 'cifar10':
    netloss = LossNetwork()

if opt.cuda:
    netAttacker.cuda()
    netClassifier.cuda()
    netloss.cuda()

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

# setup optimizer
optimizerAttacker = optim.Adam(netAttacker.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.l2reg)

# Important pre-set noise variable shape：[batchsize,3,imageSize,imageSize] fixed noise for universal perturbatio
noise_data = np.random.uniform(0, 0.5, opt.imageSize * opt.imageSize * 3)
if opt.mode == 'train':
    np.savetxt(opt.outf + '/U_input_noise.txt', noise_data)
else:
    if(os.path.exists(opt.outf + '/U_input_noise.txt') == False):
        np.savetxt(opt.outf + '/U_input_noise.txt', noise_data)
    noise_data = np.loadtxt(opt.outf + '/U_input_noise.txt')

# random noise of train and test
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)

if opt.cuda:
    noise = noise.cuda()
noise = Variable(noise)


def test(epoch, c,noise):
    """
    Parameters: epoch          -- 训练轮数
                shrink         -- 扰动缩放系数
                noise          -- 输入的随机噪声
    """
    netAttacker.eval()
    netClassifier.eval()
    L_inf = []
    L2 = []
    pert_norm = []
    dist = []
    adv_norm = []
    non_adv_norm = []
    total_count = 0
    success_count = 0
    skipped = 0
    no_skipped = 0
    with torch.no_grad():
        for batch_idx, (inputv, cls) in enumerate(test_loader):
            if opt.cuda:
                inputv = inputv.cuda()
            inputv = Variable(inputv)
            batch_size = inputv.size(0)

            targets = torch.LongTensor(batch_size)
            if opt.cuda:
                targets = targets.cuda()
                cls = cls.cuda()
            targets = Variable(targets)

            prediction = netClassifier(inputv)

            if opt.restrict_to_correct_preds == 1:
                incorrect_idxs = np.array(np.where(prediction.data.max(1)[1].eq(cls).cpu().numpy() == 0))[0].astype(int)
                skipped += incorrect_idxs.shape[0]
                no_skipped += (batch_size - incorrect_idxs.shape[0])
                if incorrect_idxs.shape[0] == batch_size:
                    print("All original predictions were incorrect! Skipping batch!")
                    continue
                elif incorrect_idxs.shape[0] > 0 and incorrect_idxs.shape[0] < batch_size:
                    correct_idxs = np.setdiff1d(np.arange(batch_size), incorrect_idxs)
                    correct_idxs = torch.LongTensor(correct_idxs)
                    if opt.cuda:
                        correct_idxs = correct_idxs.cuda()
                    inputv = torch.index_select(inputv, 0, Variable(correct_idxs))
                    prediction = torch.index_select(prediction, 0, Variable(correct_idxs))
                    cls = torch.index_select(cls, 0, correct_idxs)
            # remove samples that are of the target class
            if opt.targeted == 1:
                targets.data.resize_as_(cls).fill_(opt.chosen_target_class)
                ids = np.array(np.where(targets.data.eq(cls).cpu().numpy() == 0))[0].astype(int)
                ids = torch.LongTensor(ids)
                if opt.cuda:
                    ids = ids.cuda()
                inputv = torch.index_select(inputv, 0, Variable(ids))
                prediction = torch.index_select(prediction, 0, Variable(ids))
                cls = torch.index_select(cls, 0, ids)

            batch_size = inputv.size(0)
            targets.resize_(batch_size)
            noise.resize_(batch_size, opt.nz, 1, 1).normal_(0, 0.5)

            # compute an adversarial example and its prediction
            prediction = netClassifier(inputv)

            ## generate per image perturbation from fixed noise
            if opt.perturbation_type == 'universal':
                delta = netAttacker(noise)
            else:
                delta = netAttacker(inputv)

            adv_sample_ = torch.add(delta * c , inputv)
            # adv_sample_ = delta * c + inputv

            # 参考GAP do clamping per channel
            for cii in range(3):
                adv_sample_.data[:, cii, :, :] = adv_sample_.data[:, cii, :, :].clamp(inputv.data[:, cii, :, :].min(),
                                                                                      inputv.data[:, cii, :, :].max())

            # adv_sample = torch.clamp(adv_sample_, min_val, max_val)
            adv_sample = adv_sample_
            save_adv_sample = adv_sample.clone()
            # ######################  save adv_sample #########################
            if opt.save_adv == 1:
                for i in range(0, batch_size):
                    save_adv = save_adv_sample[i]
                    # undo normalize image color channels
                    for c2 in range(3):
                        save_adv.data[c2, :, :] = (save_adv.data[c2, :, :] * stddev_arr[c2]) + mean_arr[c2]
                    # save_adv = rescale(save_adv, mean, std)
                    label_path = str(int(cls[i]))
                    if not os.path.exists(os.path.join(opt.outf,label_path)):
                        os.mkdir(os.path.join(opt.outf,label_path))
                    vutils.save_image(save_adv.data, '{}/{}/{}_{}.png'.format(opt.outf, label_path, batch_idx,i))  # 保存3D 0-1 tensor 为图片格式
            # ######################  End save adv_sample #########################

            adv_prediction = netClassifier(adv_sample)

            if opt.targeted == 1:
                no_idx = np.array(np.where(adv_prediction.data.max(1)[1].eq(targets.data).cpu().numpy() == 0))[0].astype(
                    int)
            else:
                no_idx = np.array(np.where(adv_prediction.data.max(1)[1].eq(prediction.data.max(1)[1]).cpu().numpy() == 1))[
                    0].astype(int)

            success_count += inputv.size(0) - len(no_idx)
            total_count += inputv.size(0)
            if len(no_idx) != inputv.size(0):
                yes_idx = np.setdiff1d(np.array(range(inputv.size(0))), no_idx)
                for i, adv_idx in enumerate(yes_idx):
                    clean = inputv[adv_idx].data.view(1, nc, opt.imageSize, opt.imageSize)
                    adv = adv_sample[adv_idx].data.view(1, nc, opt.imageSize, opt.imageSize)
                    pert = (inputv[adv_idx] - adv_sample[adv_idx]).data.view(1, nc, opt.imageSize, opt.imageSize)

                    if opt.dataset == 'cifar10':
                        adv_ = rescale(adv_sample[adv_idx], mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                        clean_ = rescale(inputv[adv_idx], mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                    elif opt.dataset == 'ImageNet':
                        adv_ = rescale(adv_sample[adv_idx], mean=netClassifier.mean, std=netClassifier.std)
                        clean_ = rescale(inputv[adv_idx], mean=netClassifier.mean, std=netClassifier.std)

                    linf = torch.max(torch.abs(adv_ - clean_)).data.cpu().numpy()
                    noise_norm = torch.sqrt(torch.sum((clean_[:, :, :]*255 - adv_[:, :, :]*255) ** 2)).data.cpu().numpy()
                    image_norm = torch.sqrt(torch.sum(clean_[:, :, :] ** 2)).data.cpu().numpy()
                    adv_norm_s = torch.sqrt(torch.sum(adv_[:, :, :] ** 2)).data.cpu().numpy()

                    dist.append(noise_norm / image_norm)
                    pert_norm.append(noise_norm)
                    non_adv_norm.append(image_norm)
                    adv_norm.append(adv_norm_s)
                    L_inf.append(linf)

            progress_bar(batch_idx, len(test_loader),
                         "Val E%s, A_Succ %.5f L_inf %.5f L2 %.5f (Pert %.2f, Adv %.2f, Clean %.2f) C %.6f Skipped %.1f%%" % (
                         epoch, success_count / total_count, np.mean(L_inf), np.mean(dist), np.mean(pert_norm),
                         np.mean(adv_norm), np.mean(non_adv_norm), c, 100 * (skipped / (skipped + no_skipped))))
            WriteToFile('./%s/train_log' % (opt.outf),
                        "Val Epoch %s batch_idx %s A_Succ %.5f L_inf %.5f L2 %.5f (Pert %.2f, Adv %.2f, Clean %.2f) C %.6f Skipped %.1f%%" % (
                        epoch, batch_idx, success_count / total_count, np.mean(L_inf), np.mean(dist), np.mean(pert_norm),
                        np.mean(adv_norm), np.mean(non_adv_norm), c, 100 * (skipped / (skipped + no_skipped))))





if __name__ == '__main__':

    c = opt.shrink
    if(opt.mode == 'train'):
        print("please set the mode is test",target_model)
    else:
        start = time.time()
        test(1, c, noise)
        end = time.time()
