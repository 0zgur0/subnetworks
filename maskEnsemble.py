'''Train CIFAR10 with PyTorch.'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random

from models.maskEnsemble_vgg import vgg_maskEnsemble
from models.maskEnsemble_resnet import ResNet18_maskEnsemble, ResNet34_maskEnsemble, ResNet50_maskEnsemble

from utils_uncertanity import _ECELoss, Entropy, function_space_analysis, AUROC


def seed_everything(seed: int):
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('seeding everything w/ seed', seed)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--dataset', default='Cifar10', type=str, help='Cifar10/Cifar100/ImageNet')
parser.add_argument('--datadir', default='/cluster/scratch/tmehmet/cifar10/', type=str, help='dataset directory')
parser.add_argument('--ood_datadir', default='/cluster/scratch/tmehmet/svhn/', type=str, help='dataset directory')
parser.add_argument('--ensemble', '-e', default=4, type=int, help='number of ensemble members')
parser.add_argument('--scale', default=2., type=float, help='scale factor of maskEnsmble')
parser.add_argument('--OOD', default=False, type=bool, help='Out-Of-Distribution (OOD) evaluation')
parser.add_argument('--OOD-rot-gain', default=180., type=float, help='max rotation for OOD')
parser.add_argument('--net-type', default='Resnet18', type=str, help='net type: VGG11-13-16-19, wideResnet, Resnet18-34-50')
parser.add_argument('--wR-depth', default=18, type=int, help='wideResnet depth')
parser.add_argument('--wR-widen-factor', default=1, type=int, help='wideResnet widen factor')
parser.add_argument('--max-epoch', default=200, type=int, help='number of epochs to train')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--optim-config', default='A', type=str, help='optimizer configuration: A, B')
parser.add_argument('--grad-clip', default=5, type=float, help='max gradient value allowed')
parser.add_argument('--measure_time', default=True, type=bool, help='measure inference time')
parser.add_argument('--eval', default=False, type=bool, help='mode: eval or train')
parser.add_argument('--resume', '-r', default=None, type=str, help='resume from checkpoint')
parser.add_argument('--save-model', default=False, type=bool, help='where the trained model is saved')
parser.add_argument('--save-dir', '-s', type=str, help='resume from checkpoint',
                    default='/cluster/scratch/tmehmet/checkpoints/')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--num-worker', default=4, type=int, help='number of workers')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_ensembles = args.ensemble

#-----------------------Data transforms -------------------------------------
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.OOD:
    transform_test = transforms.Compose([
        transforms.RandomRotation(degrees=args.OOD_rot_gain),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

else:
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

#-------------------------------Datasets----------------------------
if args.dataset=='Cifar10':
    trainset = torchvision.datasets.CIFAR10(root=args.datadir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.datadir, train=False, download=True, transform=transform_test)
    num_classes = 10

elif args.dataset=='Cifar100':
    trainset = torchvision.datasets.CIFAR100(root=args.datadir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.datadir, train=False, download=True, transform=transform_test)
    num_classes = 100

#----------------------------Dataloaders---------------------------
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=args.num_worker)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.num_worker)

#------------------------------OOD dataset/loader-----------------------------
ood_transform = transforms.Compose([
    transforms.Scale(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
svhn = torchvision.datasets.SVHN(root=args.ood_datadir, download=True, transform=ood_transform, split='test')
ood_dataset_loader = torch.utils.data.DataLoader(dataset=svhn, batch_size=1, shuffle=False, num_workers=args.num_worker)

#---------------------------------Model---------------------------
print('==> Building model..')

if args.net_type in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    net = vgg_maskEnsemble(cfg=cfg[args.net_type], pretrained=False,  n=num_ensembles, scale=args.scale, num_classes=num_classes)

elif args.net_type == 'Resnet18':
    net = ResNet18_maskEnsemble(n=num_ensembles, scale=args.scale, num_classes=num_classes)
elif args.net_type == 'Resnet34':
    net = ResNet34_maskEnsemble(n=num_ensembles, scale=args.scale, num_classes=num_classes)
elif args.net_type == 'Resnet50':
    net = ResNet50_maskEnsemble(n=num_ensembles, scale=args.scale, num_classes=num_classes)


net = net.to(device)
print(net)

model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Number of model parameters: ', params)

if args.eval and args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)


# Uncertanity criteria
if device == 'cuda':
    ece_criterion = _ECELoss().cuda()
    entropy_criterion = Entropy().cuda()
    AUROC_criterion = AUROC().cuda()
else:
    ece_criterion = _ECELoss()
    entropy_criterion = Entropy()
    AUROC_criterion = AUROC()

# Subnetwork independency analysis
indAnaysisF = function_space_analysis()


# Training
def train(net, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Train: Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return net


# Testing
def test(net, loader, epoch, num_used_members, ood=False):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    outputs_all = list()
    outputs_all_mean = list()
    targets_all = list()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # maskEnsemble - repeat the input in the batch dim
            inputs = inputs.repeat(num_ensembles, 1, 1, 1)

            outputs = net(inputs)

            # maskEnsemble - Average ensemble members softmax output
            outputs_mean = torch.mean(outputs, dim=0, keepdim=True)

            loss = criterion(outputs_mean, targets)

            outputs_all.append(outputs.view(1, outputs.shape[0],outputs.shape[1]))
            outputs_all_mean.append(outputs_mean)
            targets_all.append(targets)

            test_loss += loss.item()
            _, predicted = outputs_mean.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        outputs_all = torch.cat(outputs_all)
        outputs_all = outputs_all.permute(0,2,1)
        outputs_all_mean = torch.cat(outputs_all_mean)
        targets_all = torch.cat(targets_all)

        # Calibration performances
        ece, accs, confs = ece_criterion(outputs_all_mean, targets_all)
        entropy = entropy_criterion(outputs_all_mean)

        if not ood:
            print('Val: Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            print('ECE: {:.4f}, Entropy: {:.4f}'.format(ece.item(), entropy.item()))

            if num_used_members > 1:
                disagreement, distance = indAnaysisF(outputs_all[..., 0], outputs_all[..., 1])
                print('disagreement (%): {:.4f}, distance (KL-div): {:.4f}'.format(disagreement.item() * 100,
                                                                                   distance.item()))

            if args.save_model:
                # Save checkpoint.
                acc = 100. * correct / total
                if acc > best_acc:
                    print('Saving the model here: ', args.save_dir)
                    state = {
                        'net': net.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    if not os.path.isdir(args.save_dir):
                        os.mkdir(args.save_dir)
                    torch.save(state, os.path.join(args.save_dir,
                                                    'ckpt_bestVal_maskEnsemble_dataset{}_e{}_scale{}_net{}_seed{}.pth'
                                                   .format(args.dataset, args.ensemble, args.scale,
                                                           args.net_type, args.seed)))
                    best_acc = acc
                elif epoch == args.max_epoch - 1:
                    state = {
                        'net': net.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    torch.save(state,
                               os.path.join(args.save_dir,
                                              'ckpt_bestVal_maskEnsemble_dataset{}_e{}_scale{}_net{}_seed{}.pth'
                                                   .format(args.dataset, args.ensemble, args.scale,
                                                           args.net_type, args.seed)))

    return outputs_all_mean, targets_all


# -------------MAIN---------------
seed_everything(args.seed)
print(args)

if args.eval:
    test(net, 0)
else:
    for epoch in range(start_epoch, start_epoch + args.max_epoch):

        net = train(net, epoch)
        scheduler.step()

        if True: #epoch > int(args.max_epoch * 0.75):
            id_scores, _ = test(net, testloader, epoch, num_used_members=args.ensemble)
            ood_scores, _ = test(net, ood_dataset_loader, epoch, num_used_members=args.ensemble, ood=True)
            auroc = AUROC_criterion(id_scores, ood_scores)

