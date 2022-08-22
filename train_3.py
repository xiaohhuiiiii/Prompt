
import torch
import torchvision
import torchvision.transforms as tf
import torchvision.models as models
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import timm
from utils import AverageMeter, accuracy, ProgressMeter, Metrics
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import config
import time
import wandb
import os
from os.path import join
import shutil
import torch.nn as nn
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
from add_prompt import Add_Prompt


# os.environ['TORCH_HOME'] = '/data16/xiaohui/torch_model'
torch.set_num_threads(4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    # define transforms and param
    train_param = config.train_param
    val_param = config.val_param
    train_transforms = tf.Compose([tf.Resize(256), 
                                tf.RandomCrop(224), 
                                tf.ToTensor(), 
                                tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    val_transforms = tf.Compose([tf.Resize(224), 
                                tf.ToTensor(), 
                                tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    # get datasets
    train_data = CIFAR100('/data/Public/Datasets/cifar100/', transform=train_transforms,
                             download=False, train=True)
    val_data = CIFAR100('/data/Public/Datasets/cifar100/', transform=val_transforms,
                           download=False, train=False)
    train_loader = DataLoader(train_data, batch_size=train_param['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=val_param['batch_size'], shuffle=False, num_workers=4)
    # select model
    model = Add_Prompt()
    model = model.to(device)
    model = nn.DataParallel(model)
    save_path = join('/home/2021/xiaohui/Storage/Prompt/checkpoint', config.project_name)
    optimizer = torch.optim.SGD(model.parameters(), lr=train_param['lr'], weight_decay=train_param['weight_decay'])
    Scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-9)
    if train_param['loss'] == 'CE':
        if train_param['loss_weight'] is not None:
            weight = torch.tensor(train_param['loss_weight']).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = torch.nn.CrossEntropyLoss()


    # train start
    best_acc = 0
    indices = list(range(100))
    if config.use_wandb:
        wandb.init(project=config.project_name)
        wandb.config.update(config.train_param)
        wandb.run.name = 'prompt'
        wandb.watch(model, criterion, log='all', log_freq=10)

    for epoch in range(train_param['epoch']):
        model.train()
        train(model, train_loader, optimizer, criterion, epoch, train_param, indices)

        acc1 = validate(model, val_loader, criterion, val_param, epoch, indices)

        is_best = acc1 > best_acc
        best_acc = max(acc1, best_acc)
        Scheduler.step()
        if not os.path.exists(join(save_path, )):
            os.makedirs(save_path)
    print('BestAcc: {}'.format(best_acc))
    if config.if_save:
        torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc
            }, join(save_path, 'model.pth.tar'))
        
        # 测试
    wandb.run.finish()

def train(model, train_loader, optimizer, criterion, epoch, train_param, indices):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix='Epoch: [{}]'.format(epoch)
    )
    num_batches_per_epoch = len(train_loader)
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        output = output[:, indices]
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1 = accuracy(output, target, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % train_param['print_freq'] == 0:
            progress.display(i)

            if config.use_wandb:
                wandb.log({
                    'training_loss': losses.avg,
                    'training_acc': top1.avg,
                    'epoch': epoch
                })

    return losses.avg, top1.avg

def validate(model, val_loader, criterion, val_param, epoch, indices, mode='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    if mode == 'test':
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1],
            prefix='Test: ')
    else:
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1],
            prefix='Validate: ')

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            output = output[:, indices]
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % val_param['print_freq'] == 0:
                progress.display(i)

        print(' * Acc {top1.avg:.3f}'
              .format(top1=top1))

        if config.use_wandb:
            wandb.log({
                'val_loss': losses.avg,
                'val_acc': top1.avg,
                'epoch': epoch
            })

    return top1.avg

if __name__ == '__main__':
    main()