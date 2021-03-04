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
from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import time
import sys
sys.path.append('../')
from Utils.utils import *
from Classfiers.models import *
from Classfiers.models import ImageNetLossNetwork
from Attacks.Generator.resnetG import define_G

parser = argparse.ArgumentParser()
# the param of train process
parser.add_argument('--manualSeed', type=int, default=5198, help='random seed')
parser.add_argument('--workers', type=int, help='workers', default=2)
parser.add_argument('--batchSize', type=int, default=4, help='train batch size')
parser.add_argument('--testBatchSize', type=int, default=8, help='test batch size')
parser.add_argument('--epochs', type=int, default=500, help='epoch')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='the param of optim')
parser.add_argument('--l2reg', type=float, default=0.01, help='the param of optim')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--imageSize', type=int, default=224, help='the weight height of image ')
# the param of attack
parser.add_argument('--optimize_on_success', type=int, default=0, help="whether to optimize on samples that are already successful")
parser.add_argument('--targeted', type=int, default=0, help='targeted attack or not')
parser.add_argument('--chosen_target_class', type=int, default=1, help='the class to target')
parser.add_argument('--restrict_to_correct_preds', type=int, default=1,
                    help='only compute adv examples on correct predictions')
parser.add_argument('--shrink', type=float, default=0.22, help='scale perturbation ')
parser.add_argument('--shrink_inc', type=float, default=0.01,
                    help='update the scale')
parser.add_argument('--max_norm', type=float, default=0.06, help='max allowed perturbation')
parser.add_argument('--norm', type=str, default='linf', help='l2 or linf')
parser.add_argument('--perturbation_type', type=str, default='universal', help='"universal" or "imdep" (image dependent)')
parser.add_argument('--lfs_weight', type=float, default=1.0, help='weight of loss')
parser.add_argument('--ldist_weight', type=float, default= 2.0, help='weight of loss')
parser.add_argument('--content_weight', type=float, default=1, help='weight of loss')
parser.add_argument('--targetClassifier', type=str, default='incv3', help=" classifier (incv3 or densenet121 or vgg16)")
parser.add_argument('--dataset', type=str, default='ImageNet', help='the type of dataset')
parser.add_argument('--imagenetTrain', type=str, default='E:\\dataset\\ImageNet\\train', help='the path of dataset')
parser.add_argument('--imagenetVal', type=str, default='E:\\dataset\\ImageNet\\val\\val_sub', help='the path of dataset')


# the param of test mode
parser.add_argument('--netAttacker', default='', help="the path of netAttacker")
parser.add_argument('--mode', type=str, default='train', help='running mode')
parser.add_argument('--save_adv', type=int, default=0, help='save advserial example or not')
parser.add_argument('--outf', default='./advres_imagenet_incv3', help='the path of netAttacker')


opt = parser.parse_args()
target_model = opt.targetClassifier

print(opt)
torch.cuda.set_device(0)
global_acc = 0

# define normalization means and stddevs
model_dimension = 299 if opt.targetClassifier == 'incv3' else 256
center_crop = 299 if opt.targetClassifier == 'incv3' else 224
if opt.targetClassifier == 'incv3':
    opt.imageSize = 299

mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]



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
netAttacker = define_G(3, 3, 64, 'resnet_6blocks',
                          norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])
if opt.netAttacker != '':
    checkpoint = torch.load(opt.netAttacker,map_location='cuda:0')
    netAttacker.load_state_dict(checkpoint['net'])
    global_acc = checkpoint['acc']
    target_model = checkpoint['target_model']
    print("loading netAttacker chechpoint success ", checkpoint['acc'])

#print(netAttacker)
print("===> loading targetClassifier ")
if opt.targetClassifier == 'incv3':
    netClassifier = torchvision.models.inception_v3(pretrained=True)
elif opt.targetClassifier == 'vgg16':
    netClassifier = torchvision.models.vgg16(pretrained=True)
elif opt.targetClassifier == 'densenet121':
    netClassifier = torchvision.models.densenet121(pretrained=True)
elif opt.targetClassifier == "wide_resnet50_2":
    netClassifier = torchvision.models.wide_resnet50_2(pretrained=True)
elif opt.targetClassifier == "googlenet":
    netClassifier = torchvision.models.googlenet(pretrained=True)
netClassifier.volatile = True

print("===> loading lossnetwork")
if opt.dataset == 'cifar10':
    netloss = LossNetwork()
# else:
#     netloss = ImageNetLossNetwork()

if opt.cuda:
    netAttacker.cuda()
    netClassifier.cuda()
    # netloss.cuda()

print('===> loading dataset..')
if opt.dataset == 'cifar10':
    print("only support the ImageNet dataset")
elif opt.dataset == 'ImageNet':
    normalize = transforms.Normalize(mean=mean_arr,
                                     std=stddev_arr)

    data_transform = transforms.Compose([
        transforms.Resize(model_dimension),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = torchvision.datasets.ImageFolder(root = opt.imagenetTrain, transform = data_transform)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.workers, batch_size=opt.batchSize,
                                      shuffle=True)
    test_set = torchvision.datasets.ImageFolder(root=opt.imagenetVal, transform=data_transform)
    test_loader = DataLoader(dataset=test_set, num_workers=opt.workers, batch_size=opt.testBatchSize, shuffle=True)

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

# random noise of train
im_noise = np.reshape(noise_data, (3, opt.imageSize, opt.imageSize))
im_noise = im_noise[np.newaxis, :, :, :]
im_noise_tr = np.tile(im_noise, (opt.batchSize, 1, 1, 1))
noise_tensor_tr = torch.from_numpy(im_noise_tr).type(torch.FloatTensor)
noise_tr = Variable(noise_tensor_tr)
# random noise of test
im_noise_te = np.tile(im_noise, (opt.testBatchSize, 1, 1, 1))
noise_tensor_te = torch.from_numpy(im_noise_te).type(torch.FloatTensor)
noise_te = Variable(noise_tensor_te)

if opt.cuda:
    noise_tr = noise_tr.cuda()
    noise_te = noise_te.cuda()


def train(epoch, c,noise):
    """Return 攻击成功率，扰动的无穷范数，扰动的2范数
    Parameters: epoch          -- 训练轮数
                shrink         -- 扰动缩放系数
                noise          -- 输入的随机噪声
    """
    global global_acc
    netAttacker.train()
    netClassifier.eval()
    # netloss.eval()
    # cl_loss 存放制作成功样本的损失，L_inf L2 norm_Ratio raw_L2 adv_L2 分别存放扰动的无穷范数，2范数，扰动2范数与原始图像2范数的比值，原始图像的2范数，对抗样本的2范数
    c_loss, L_inf, L2, pert_norm, dist, adv_norm, non_adv_norm = [], [], [], [], [], [], []
    total_count, success_count, skipped, no_skipped = 0, 0, 0, 0

    for batch_idx, (inputv, cls) in enumerate(train_loader):
        if batch_idx > 1250 * 5:
            break
        optimizerAttacker.zero_grad()
        batch_size = inputv.size(0)
        targets = torch.LongTensor(batch_size)
        if opt.cuda:
            inputv = inputv.cuda()
            targets = targets.cuda()
            cls = cls.cuda()

        inputv = Variable(inputv)
        targets = Variable(targets)

        prediction = netClassifier(inputv)

        # 1 filter
        if opt.restrict_to_correct_preds == 1:
            # get indexes where the original predictions are incorrect
            incorrect_idxs = np.array(np.where(prediction.data.max(1)[1].eq(cls).cpu().numpy() == 0))[0].astype(int)
            skipped += incorrect_idxs.shape[0]
            no_skipped += (batch_size - incorrect_idxs.shape[0])
            if incorrect_idxs.shape[0] == batch_size:
                print("All original predictions were incorrect! Skipping batch!")
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

        # if this is a targeted attack, fill the target variable and filter out examples that are of that target class
        if opt.targeted == 1:
            targets.resize_as_(cls).fill_(opt.chosen_target_class)
            ids = np.array(np.where(targets.data.eq(cls).cpu().numpy() == 0))[0].astype(int)
            ids = torch.LongTensor(ids)
            if opt.cuda:
                ids = ids.cuda()
            inputv = torch.index_select(inputv, 0, Variable(ids))
            prediction = torch.index_select(prediction, 0, Variable(ids))
            cls = torch.index_select(cls, 0, ids)

        batch_size = inputv.size(0)
        targets.resize_(batch_size)
        noise.resize_(batch_size, 3, opt.imageSize, opt.imageSize)

        # compute an adversarial example and its prediction
        prediction = netClassifier(inputv)

        ## generate per image perturbation from fixed noise
        if opt.perturbation_type == 'universal':
            delta = netAttacker(noise)
        else:
            delta = netAttacker(inputv)

        # crop slightly to match inception
        if opt.targetClassifier == 'incv3':
            delta = nn.ConstantPad2d((0, -1, -1, 0), 0)(delta)

        adv_sample_ = torch.add(delta * c, inputv)

        # do clamping per channel
        for cii in range(3):
            adv_sample_.data[:,cii,:,:] = adv_sample_.data[:,cii,:,:].clamp(inputv.data[:,cii,:,:].min(), inputv.data[:,cii,:,:].max())

        adv_sample = adv_sample_
        adv_prediction = netClassifier(adv_sample)

        if opt.targeted == 1:
            no_idx = np.array(np.where(adv_prediction.data.max(1)[1].eq(targets.data).cpu().numpy() == 0))[0].astype(
                int)
        else:
            no_idx = np.array(np.where(adv_prediction.data.max(1)[1].eq(prediction.data.max(1)[1]).cpu().numpy() == 1))[
                0].astype(int)

        # update success and total counts
        success_count += inputv.size(0) - len(no_idx)
        total_count += inputv.size(0)

        # 2 filter
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
                    adv_ = rescale(adv_sample[adv_idx], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                    clean_ = rescale(inputv[adv_idx], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

                linf = torch.max(torch.abs(adv_ - clean_)).data.cpu().numpy()
                noise_norm = torch.sqrt(torch.sum((clean_[:, :, :] - adv_[:, :, :]) ** 2)).data.cpu().numpy()
                #noise_norm = 255*torch.dist(adv_,clean_,p=2).data.cpu().numpy()
                image_norm = torch.sqrt(torch.sum(clean_[:, :, :] ** 2)).data.cpu().numpy()
                adv_norm_s = torch.sqrt(torch.sum(adv_[:, :, :] ** 2)).data.cpu().numpy()

                dist.append(noise_norm / image_norm)
                pert_norm.append(noise_norm)
                non_adv_norm.append(image_norm)
                adv_norm.append(adv_norm_s)
                L_inf.append(linf)

                if batch_idx == 0:
                    vutils.save_image(torch.cat((clean, pert, adv)), './{}/{}_{}.png'.format(opt.outf, epoch, i),
                                      normalize=True, scale_each=True)

        # compute success loss and make sure the adv_sample
        if opt.optimize_on_success == 0:
            if len(no_idx) != 0:
                # select the non adv examples to optimise on
                no_idx = torch.LongTensor(no_idx)
                if opt.cuda:
                    no_idx = no_idx.cuda()
                no_idx = Variable(no_idx)
                inputv = torch.index_select(inputv, 0, no_idx)
                prediction = torch.index_select(prediction, 0, no_idx)
                targets = torch.index_select(targets, 0, no_idx)
                adv_prediction = torch.index_select(adv_prediction, 0, no_idx)
                delta = torch.index_select(delta, 0, no_idx)
                adv_sample = torch.index_select(adv_sample, 0, no_idx)
        elif opt.optimize_on_success == 1:
            yes_idx = np.setdiff1d(np.arange(batch_size), no_idx)
            if yes_idx.shape[0] != 0:
                adv_prediction_succ = adv_prediction[torch.LongTensor(yes_idx).cuda()]
                prediction_succ = prediction[torch.LongTensor(yes_idx).cuda()].data.max(1)[1]
                adv_prediction_succ = F.softmax(adv_prediction_succ,dim=1)
                if no_idx.shape[0] != 0:
                    adv_prediction = adv_prediction[torch.LongTensor(no_idx).cuda()]
                adv_pred_idx = torch.FloatTensor(
                    [x[prediction_succ[i]].data[0] for i, x in enumerate(adv_prediction_succ)]).cuda()
                adv_max_idx = adv_prediction_succ.data.max(1)[0]
                success_loss = -torch.mean(torch.log(adv_max_idx) - torch.log(adv_pred_idx))
            else:
                success_loss = 0

        #compute the loss and backward
        if len(no_idx) != 0:
            # compute loss and backprop
            adv_prediction_softmax = F.softmax(adv_prediction,dim=1)
            # adv_prediction_np = adv_prediction.data.cpu().numpy()
            adv_prediction_np = adv_prediction_softmax.data.cpu().numpy()
            curr_adv_label = Variable(torch.LongTensor(np.array([arr.argsort()[-1] for arr in adv_prediction_np])))
            if opt.targeted == 1:
                # targ_adv_label = Variable(
                #     torch.LongTensor(np.array([targets.data[i] for i, arr in enumerate(adv_prediction_np)])))
                targ_adv_label = Variable(
                    torch.LongTensor(np.array([arr[opt.chosen_target_class] for arr in adv_prediction_np])))
            else:
                targ_adv_label = Variable(torch.LongTensor(np.array([arr.argsort()[-2] for arr in adv_prediction_np])))
            if opt.cuda:
                curr_adv_label = curr_adv_label.cuda()
                targ_adv_label = targ_adv_label.cuda()
            curr_adv_pred = adv_prediction_softmax.gather(1, curr_adv_label.unsqueeze(1))
            targ_adv_pred = adv_prediction_softmax.gather(1, targ_adv_label.unsqueeze(1))

            if opt.optimize_on_success == 1:
                classifier_loss = opt.lfs_weight * torch.mean(torch.log(curr_adv_pred) - torch.log(targ_adv_pred)) + success_loss
            else:
                classifier_loss = opt.lfs_weight * torch.mean(torch.log(curr_adv_pred) - torch.log(targ_adv_pred))

            if opt.norm == 'linf':
                ldist_loss = opt.ldist_weight * torch.max(torch.abs(adv_sample - inputv))
            elif opt.norm == 'l2':
                ldist_loss = opt.ldist_weight * torch.mean(torch.sqrt(torch.sum((adv_sample - inputv) ** 2)))
            else:
                print("Please define a norm (l2 or linf)")
                exit()

            #
            if opt.content_weight != 0:
                # 添加content-loss损失
                # features_x = netloss(inputv)
                # features_y = netloss(adv_sample)
                # content_loss = opt.content_weight * F.mse_loss(features_y.relu2_2, features_x.relu2_2)
                # content_loss = opt.content_weight * torch.max(torch.abs(features_y.relu2_2 - features_x.relu2_2))
                loss = classifier_loss + ldist_loss #+ content_loss
            else:
                loss = classifier_loss + ldist_loss

            loss.backward()
            optimizerAttacker.step()
            c_loss.append(classifier_loss.item())
        else:
            if opt.optimize_on_success == 1:
                classifier_loss = success_loss
                c_loss.append(classifier_loss)
                classifier_loss = torch.FloatTensor([classifier_loss])
                classifier_loss = Variable(classifier_loss, requires_grad=True)
                if opt.cuda:
                    classifier_loss = classifier_loss.cuda()
                loss.backward()
                optimizerAttacker.step()
            else:
                c_loss.append(0)

        # train_log to file
        progress_bar(batch_idx, len(train_loader),
                     "Tr Epoch%s, C_L %.4f A_Succ %.4f L_inf %.4f L2 %.4f (Pert %.2f, Adv %.2f, Clean %.2f) C %.4f Skipped %.1f%%"  % (
                     epoch, np.mean(c_loss), success_count / total_count, np.mean(L_inf), np.mean(dist),
                     np.mean(pert_norm), np.mean(adv_norm), np.mean(non_adv_norm), c,
                     100 * (skipped / (skipped + no_skipped))))
        WriteToFile('./%s/train_log' % (opt.outf),
                    "Tr Epoch %s batch_idx %s C_L %.4f A_Succ %.4f L_inf %.4f L2 %.5f (Pert %.2f, Adv %.2f, Clean %.2f) C %.4f Skipped %.1f%%" % (
                    epoch, batch_idx, np.mean(c_loss), success_count / total_count, np.mean(L_inf), np.mean(dist),
                    np.mean(pert_norm), np.mean(adv_norm), np.mean(non_adv_norm), c,
                    100 * (skipped / (skipped + no_skipped))))

     # save model weights
    if global_acc < success_count / total_count:
        global_acc = success_count / total_count
        print('Saving netAttacker')
        state = {
            'net': netAttacker.state_dict(),
            'acc': global_acc,
            'target_model': target_model
        }
        torch.save(state, '%s/netAttacker_%s_%.5f.t7' % (opt.outf, epoch, global_acc))


    return success_count / total_count, np.mean(L_inf), np.mean(dist)

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
            # if batch_idx > 500:
            #     break
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
            noise.resize_(batch_size, 3, opt.imageSize, opt.imageSize)

            # compute an adversarial example and its prediction
            prediction = netClassifier(inputv)

            ## generate per image perturbation from fixed noise
            if opt.perturbation_type == 'universal':
                delta = netAttacker(noise)
            else:
                delta = netAttacker(inputv)

            # crop slightly to match inception
            if opt.targetClassifier == 'incv3':
                delta = nn.ConstantPad2d((0, -1, -1, 0), 0)(delta)

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
                        adv_ = rescale(adv_sample[adv_idx], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        clean_ = rescale(inputv[adv_idx], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

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
        print("start training of attack target model :",target_model)
        for epoch in range(1, opt.epochs + 1):
            start = time.time()
            score, linf, l2 = train(epoch, c ,noise_tr)
            if linf > opt.max_norm:
                break
            # if l2 > opt.max_norm:
            #     break
            end = time.time()
            if score >= 1.00:
                break
            if epoch == 1:
                curr_pred = score
            if epoch % 2 == 0:
                prev_pred = curr_pred
                curr_pred = score
            if epoch > 2:
                if (prev_pred - curr_pred) >= 0:
                    c += opt.shrink_inc
            print("%.2f" % (c))
        print("start testing on testset :",target_model)
        test(1, c ,noise_te)
    else:
        print("start testing on testset :", target_model)
        start = time.time()
        test(1, c, noise_te)
        end = time.time()
