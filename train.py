from __future__ import print_function, division
import argparse
import json
import os
import pdb
import sys
import time
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from reid_utils import RandomIdentitySampler, logging, RandomErasing
from reid_utils.test_utils import *
from models.AN_EvReId_model import EvReId
from models.Sparsemodel import ResidualNetPTQ
from data.dataloader import *
from reid_losses import TripletLoss2
from torchmetrics.functional import structural_similarity_index_measure
from torch.utils.tensorboard import SummaryWriter
from e2vid_utils.e2vid_image_utils import *
from reid_utils import IterLoader
from evaluator import evaluator
import random


def train_model(model_ReId, optimizer, scheduler, device, num_epochs=50):  #baseline
    start_time = time.time()
    writer = SummaryWriter()
    dataloaders, dataset_sizes_allSample = load_train_data(opt)
    dataloaders_test, image_datasets_test = load_test_data(opt)
    print("data size", dataset_sizes_allSample)#how many batches
    batch_count = 0
    bestmap = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train']:
            save_network(model_ReId, epoch)

            if opt.ReId_loss == 'softmax+triplet':
                adjust_lr_softmax(optimizer, epoch)

            model_ReId.train(True)  # Set model to training mode
            running_loss = 0.0
            running_correct_category = 0
            dataloaders.new_epoch()
            for i in range(200):
                data = dataloaders.next()
                input_voxel, labels = data
                input_voxel = Variable(input_voxel.to(device))
                labels = Variable(labels.to(device))

                optimizer.zero_grad()
                
                """ReId model training"""
                [category, feature, _] = model_ReId(input_voxel)#feature:embed_feature


                """Event ReId Losses"""
                _, category_preds = torch.max(category.data, 1)

                if opt.ReId_loss == 'softmax+triplet':
                    loss_softmax = criterion_softmax(category, labels)
                    loss_triplet, _, _ = criterion_triplet(feature, labels)
                    # loss_triplet = torch.tensor([0.]).to(device)
                    loss_reid = loss_softmax + loss_triplet

                # writer.add_scalars(f'batch_loss', {'reid_loss': loss_reid}, batch_count)
                print('batch loss', loss_softmax.item(), loss_triplet.item())
                batch_count += 1

                """Total Loss"""
                loss = loss_reid
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item()

                category_preds = category.data.max(1)[1]
                running_correct_category += torch.sum(category_preds == labels.data)

            epoch_loss = running_loss / 200
            epoch_acc = running_correct_category.cpu().numpy() / 200 / opt.batchsize
            print('{} Loss: {:.4f} Acc_category: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            mapp, map_embed = evaluator(opt, model_ReId, dataloaders=dataloaders_test, image_datasets=image_datasets_test)

            is_best = (map_embed > bestmap)
            bestmap = max(map_embed, bestmap)
            if is_best:
                save_network(model_ReId, 'best')

            print('{} evaluate_map: {:.4f} evaluate_mapembed: {:.4f}'.format(phase, mapp, map_embed))

            """tensorboard"""
            writer.add_scalars(main_tag='training', tag_scalar_dict={'loss':epoch_loss, 'accuracy':epoch_acc, 'map':mapp, 'mapembed':map_embed}, global_step=epoch)

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 30, time_elapsed % 30))
    writer.close()
    return model_ReId
    

def save_network(network1, epoch_label): 
    save_filename = 'net_%s.pth'% epoch_label
    save_path1 = os.path.join('./' + opt.file_name, save_filename)
    torch.save(network1.cpu().state_dict(), save_path1)
    if torch.cuda.is_available():
        network1.cuda(0)


def adjust_lr_softmax(optimizer, ep):
    if ep < 40:
        lr = 0.01
    elif ep < 50:
        lr = 0.005
    else:
        lr = 0.0001
    for index in range(len(optimizer.param_groups)):
        if index == 0:
            optimizer.param_groups[index]['lr'] = lr * 0.1
        else:
            optimizer.param_groups[index]['lr'] = lr


def load_test_data(opt):

    if opt.represent == "voxel" and opt.raw == 1:
        image_datasets = {x: voxelDataset(mode=x, arg=opt) for x in ['gallery', 'query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                      shuffle=False, num_workers=8) for x in ['gallery', 'query']}
        return dataloaders, image_datasets


def load_train_data(opt):

    """Load Train Data"""
    if opt.represent == "voxel":

        if opt.ReId_loss == 'softmax+triplet' and opt.raw == 1:
            cls_datasets = voxelDataset(mode='train', arg=opt)
            cls_loader = IterLoader(torch.utils.data.DataLoader(cls_datasets,
                                                                sampler=RandomIdentitySampler(cls_datasets,
                                                                                              opt.num_instances),
                                                                batch_size=opt.batchsize, shuffle=False,
                                                                num_workers=opt.num_workers, drop_last=True),
                                    length=opt.iter)

            dataset_sizes_allSample = len(cls_loader)
        print("size of all samples", dataset_sizes_allSample)  # how much bathes
        return cls_loader, dataset_sizes_allSample


def set_optimizer(model):
    ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = torch.optim.SGD([
                     {'params': base_params},
                     {'params': model.model.fc.parameters()},
                     {'params': model.classifier.parameters()}
                     ], lr=0.001, weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
    return optimizer


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    """Parser"""
    parser = argparse.ArgumentParser(description='Event_ReId_denoising')
    parser.add_argument('--represent', default='voxel', type=str, help='representation of events for reid')
    parser.add_argument('--gpu_ids', default='0', type=str,help='gpu_ids: e.g. 0  0,1')
    parser.add_argument('--name', default='Dnoise_Event_ReId', type=str, help='output model name')
    parser.add_argument('--num_ids', default=22, type=int, help='number of identities')
    parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
    parser.add_argument('--ReId_loss', default='softmax+triplet', type=str, help='choice of reid loss')
    parser.add_argument('--num_instances', default=8, type=int, help='for triplet loss')
    parser.add_argument('--margin', default=4, type=float, help='triplet loss margin')
    parser.add_argument('--num_Bin', default=5, type=int, help='number of channels of spatiotemporal event-voxel')
    parser.add_argument('--epoch', default=60, type=int, help='training epoch')
    parser.add_argument('--file_name', default='training', type=str, help='file name to save weights and log file')
    parser.add_argument('--iter', default=200, type=int, help='iteration of each batch')
    parser.add_argument('--num_workers', default=8, type=int, help='num workers for dataloader')

    "IF train with denoise"
    parser.add_argument('--raw', default=1, type=int, help='if 0, use denoise')

    "optimizer"
    parser.add_argument('--lr', type=float, default=0.05,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    opt = parser.parse_args()

    """Save Log History"""
    sys.stdout = logging.Logger(os.path.join(opt.file_name +'/'+opt.name+'/', 'log.txt'))

    """Set GPU"""
    gpu_ids = []
    str_gpu_ids = opt.gpu_ids.split(',')
    for str_id in str_gpu_ids:
        gpu_ids.append(int(str_id))
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    cudnn.enabled = True
    cudnn.benchmark = True

    "确保可复现"
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)  # 为CPU设置随机种子
    torch.cuda.manual_seed(123)  # 为当前GPU设置随机种子

    """Person ReId Model"""
    reid_model = EvReId(class_num=opt.num_ids, num_channel=opt.num_Bin) #模型改了记得改回来

    """Optimizer"""
    optimizer = set_optimizer(reid_model)

    reid_model = reid_model.to(device)

    """Set ReId Loss function"""
    criterion_triplet = TripletLoss2(opt.margin)
    criterion_softmax = nn.CrossEntropyLoss()#CE就是softmax，要避免出现log0

    """Save Dir"""
    dir_name = os.path.join('./' + opt.file_name, opt.name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    # save opts
    with open('%s/opts.json'%dir_name, 'w') as fp:
        json.dump(vars(opt), fp, indent=1)

    """Start Training"""
    model_ = train_model(reid_model, optimizer, scheduler=None, device=device, num_epochs=opt.epoch)
