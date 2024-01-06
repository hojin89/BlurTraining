import os
import sys
import random
import time
import numpy
import scipy.io
import matplotlib.pyplot as plt
import collections
import kornia

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets

from torchsummary import summary
from numpy.random import choice
from v1_utils import *

sys.path.insert(1, '/home/tonglab/Hojin/Project/PycharmProjects/CORnet_Pilot/')
from cornet.cornet_s import *

def main():

    #### Parameters ####################################################################################################
    model_path = '/hdd/Project/PycharmProjects/Blur2/v6/1_AlexNet_Epoch70_WD1e-4_LR1e-3'
    # model_path = '/hdd/Project/PycharmProjects/Blur2/v6/2_VGG19_Epoch70_WD1e-4_LR1e-3'
    # model_path = '/hdd/Project/PycharmProjects/Blur2/v6/3_ResNet50_Epoch70_WD1e-4_LR1e-3'
    # model_path = '/hdd/Project/PycharmProjects/Blur2/v6/4_VGG16_Epoch70_WD1e-4_LR1e-3'
    # model_path = '/hdd/Project/PycharmProjects/Blur2/v6/5_GoogLeNet_Epoch70_WD1e-4_LR1e-3'
    # model_path = '/hdd/Project/PycharmProjects/Blur2/v6/6_Inception3_Epoch70_WD1e-4_LR1e-3'
    # model_path = '/hdd/Project/PycharmProjects/Blur2/v6/7_ResNet18_Epoch70_WD1e-4_LR1e-3'
    # model_path = '/hdd/Project/PycharmProjects/Blur2/v6/8_ResNet101_Epoch70_WD1e-4_LR1e-3'
    # model_path = '/hdd/Project/PycharmProjects/Blur2/v6/9_CORnetS_Epoch70_WD1e-4_LR1e-3'
    # model_path = '/hdd/Project/PycharmProjects/Blur2/v6/10_CORnetSFF_Epoch70_WD1e-4_LR1e-3'
    # model_path = '/hdd/Project/PycharmProjects/Blur2/v6/11_CORnetSFF2_Epoch70_WD1e-4_LR1e-3'
    # model_path = '/media/tonglab/789aa685-f3c0-4b78-afb5-88d669eb7059/Project/PycharmProjects/Blur2/v6/12_ViTB16_Epoch70_WD1e-4_LR1e-3'

    num_categories = 1000
    train_batch_size = 64
    val_batch_size = 32
    start_epoch = 0
    num_epochs = 70 # 100
    save_every_epoch = 10
    initial_learning_rate = 1e-3 # SGD, 0.01; Adam, 0.0001
    gpu_ids = [1]

    #### Create/Load model #############################################################################################
    # 1. If pre-trained models used without pre-trained weights. e.g., model = models.vgg19()
    # 2. If pre-trained models used with pre-trained weights. e.g., model = models.vgg19(pretrained=True)
    # 3. If our models used.
    ####################################################################################################################

    model = models.alexnet(pretrained=False)
    # model = models.vgg16(pretrained=False)
    # model = models.vgg19(pretrained=False)
    # model = models.resnet18(pretrained=False)
    # model = models.resnet50(pretrained=False)
    # model = models.resnet101(pretrained=False)
    # model = models.googlenet(pretrained=False)
    # model = models.inception_v3(pretrained=False)
    # model = CORnet_S()
    # model = CORnet_S_FF()
    # model = CORnet_S_FF2()
    # model = models.vit_b_16(pretrained=False)

    model_name = model.__class__.__name__

    if model_name == 'Inception3':
        input_size = (3, 299, 299)
    else:
        input_size = (3, 224, 224)

    # model.cuda()
    # summary(model, (1, 224, 224))

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
    elif len(gpu_ids) == 1:
        device = torch.device('cuda:%d'%(gpu_ids[0]))
        torch.cuda.set_device(device)
        model.cuda()
        model.to(device)

    loss_function = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=1e-4)

    #### Resume from checkpoint
    try:
        os.mkdir(model_path)
    except:
        pass

    load_path = os.path.join(model_path, 'checkpoint.pth.tar')
    if os.path.isfile(load_path):
        print("... Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        start_epoch = checkpoint['epoch'] + 1

        if len(gpu_ids) <= 1: # 1. Multi-GPU to single-GPU or -CPU
            new_state_dict = collections.OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
            checkpoint['state_dict'] = new_state_dict
        else: # 2. single-GPU or -CPU to Multi-GPU
            new_state_dict = collections.OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if 'module.' in k:
                    name = k
                else:
                    name = 'module.' + k
                new_state_dict[name] = v
            checkpoint['state_dict'] = new_state_dict

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param in optimizer.param_groups:
            param['initial_lr'] = initial_learning_rate
    else:
        print("... No checkpoint found at '{}'".format(load_path))

    #### Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1, last_epoch=start_epoch-1)
    lr_scheduler.step()

    #### Data loader ###################################################################################################
    train_dataset = torchvision.datasets.ImageFolder(
        "/home/tonglab/Datasets/ILSVRC2012/train",
        # "/home/sangwon/Documents/Data/ILSVRC2012/2012_train",
        transforms.Compose([
            transforms.RandomRotation(10),
            # transforms.RandomResizedCrop(224),
            transforms.Resize(input_size[1:]),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.449], std=[0.226]) # grayscale
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # rgb
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # val_dataset = torchvision.datasets.ImageFolder(
    val_dataset = ImageFolderWithPaths(
        "/home/tonglab/Datasets/ILSVRC2012/val",
        # "/home/sangwon/Documents/Data/ILSVRC2012/2012_val",
        transforms.Compose([
            transforms.Resize(input_size[1:]),
            transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.449], std=[0.226]) # grayscale
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # rgb
        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    #### Train/Val #####################################################################################################
    for epoch in range(start_epoch, num_epochs):

        print("... Start epoch at '{}'".format(epoch))
        stat_file = open(os.path.join(model_path, 'training_stats.txt'), 'a+')

        train(train_loader, model, loss_function, optimizer, epoch, stat_file, gpu_ids)
        val(val_loader, model, loss_function, optimizer, epoch, stat_file, gpu_ids)

        if epoch < num_epochs-1:
            lr_scheduler.step()

        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_path, 'checkpoint.pth.tar'))
        if numpy.mod(epoch, save_every_epoch) == save_every_epoch-1:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_path, 'checkpoint_epoch_%d.pth.tar'%(epoch)))

def train(train_loader, model, loss_function, optimizer, epoch, stat_file, gpu_ids):

    if len(gpu_ids) > 1:
        model_name = model.module.__class__.__name__
    else:
        model_name = model.__class__.__name__

    model.train()
    loss_sum, num_correct1, num_correct5, batch_size_sum = 0., 0., 0., 0. # Accuracy

    for batch_index, (inputs, targets) in enumerate(train_loader):

        if len(gpu_ids) >= 1:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        #### Add blur with weighted random sample
        inputs = add_blur_with(inputs, [0], [1])

        if model_name == 'GoogLeNet':
            outputs, aux1, aux2 = model(inputs)
            loss1 = loss_function(outputs, targets)
            loss2 = loss_function(aux1, targets)
            loss3 = loss_function(aux2, targets)
            loss = loss1 + 0.3 * (loss2 + loss3)
        elif model_name == 'Inception3':
            outputs, aux = model(inputs)
            loss1 = loss_function(outputs, targets)
            loss2 = loss_function(aux, targets)
            loss = loss1 + 0.4 * loss2
        else:
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

        _, num_correct1_batch = is_correct(outputs, targets, topk=1)
        _, num_correct5_batch = is_correct(outputs, targets, topk=5)

        #### Accuracy
        loss_sum += loss.item()
        num_correct1 += num_correct1_batch.item()
        num_correct5 += num_correct5_batch.item()
        batch_size_sum += targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stat_str = '[Train] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f}) Accuracy5 ({:.4f})'.\
            format(epoch, optimizer.param_groups[0]['lr'], loss_sum / (batch_index + 1), num_correct1 / batch_size_sum, num_correct5 / batch_size_sum)
        progress_bar(batch_index, len(train_loader.dataset) / inputs.size(0), stat_str)

    stat_file.write(stat_str + '\n')

def val(val_loader, model, loss_function, optimizer, epoch, stat_file, gpu_ids):

    model.eval()
    loss_sum, num_correct1, num_correct5, batch_size_sum = 0., 0., 0., 0. # Accuracy
    correct1, correct5 = [0] * len(val_loader.dataset.samples), [0] * len(val_loader.dataset.samples) # Correct

    for batch_index, (inputs, targets, paths, indices) in enumerate(val_loader):

        if len(gpu_ids) >= 1:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        inputs = add_blur_with(inputs, [0], [1])

        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        correct1_batch, num_correct1_batch = is_correct(outputs, targets, topk=1)
        correct5_batch, num_correct5_batch = is_correct(outputs, targets, topk=5)

        #### Accuracy
        loss_sum += loss.item()
        num_correct1 += num_correct1_batch.item()
        num_correct5 += num_correct5_batch.item()
        batch_size_sum += targets.size(0)

        #### Correct
        for i, index in enumerate(indices):
            correct1[index] = correct1_batch.view(-1)[i].item()
            correct5[index] = torch.any(correct5_batch, dim=0).view(-1)[i].item() # top5 glitch

        stat_str = '[Validation] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f}) Accuracy5 ({:.4f})'.\
            format(epoch, optimizer.param_groups[0]['lr'], loss_sum / (batch_index + 1), num_correct1 / batch_size_sum, num_correct5 / batch_size_sum)
        progress_bar(batch_index, len(val_loader.dataset) / inputs.size(0), stat_str)

    stat_file.write(stat_str + '\n')

def is_correct(output, target, topk=1):
    with torch.no_grad():
        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        num_correct = correct[:topk].contiguous().view(-1).float().sum(0, keepdim=True) # top5 glitch

        return correct, num_correct

def add_blur_with(images, sigmas, weights):
    blurred_images = torch.zeros_like(images)
    normalize = transforms.Normalize(mean=[0.449], std=[0.226]) # grayscale
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # rgb

    for i in range(images.size(0)): # Batch size
        image = images[i, :, :, :]

        weights = numpy.asarray(weights).astype('float64')
        weights = weights / numpy.sum(weights)
        sigma = choice(sigmas, 1, p=weights)[0]
        kernel_size = 2 * math.ceil(2.0 * sigma) + 1

        if sigma == 0:
            blurred_image = image
        else:
            blurred_image = kornia.gaussian_blur2d(torch.unsqueeze(image, dim=0), kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[0, :, :, :]
        blurred_image = normalize(blurred_image)
        blurred_images[i] = blurred_image

        # fig, axes = plt.subplots(1,1)
        # axes[0].imshow(blurred_image.cpu().squeeze()) # Grayscale
        # plt.show()

    blurred_images = blurred_images.repeat(1, 3, 1, 1) # Grayscale to RGB
    return blurred_images

if __name__ == '__main__':
    main()
