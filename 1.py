from __future__ import print_function, division
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import json
import os
import pdb
import sys
import scipy.io
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from reid_utils import RandomIdentitySampler, logging, RandomErasing
from reid_utils.test_utils import *
from models.AN_EvReId_model import EvReId
from data.dataloader import *

def test_reid():
    return

def load_network(network, path):
    pretrained_dict = torch.load(path)
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    """Parser"""
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--model_path', default='training/net_best.pth', type=str, help='path to pretrained Event-ReId model wieghts')
    parser.add_argument('--represent', default='voxel', type=str, help='representation of events for reid')
    parser.add_argument('--An_model_block', default=False, help='set True, if implement Event-voxel Anonymization Block')
    parser.add_argument('--gpu_ids', default='0', type=str,help='gpu_ids: e.g. 0  0,1')
    parser.add_argument('--name', default='Denoise_Event_ReId', type=str, help='output model name')
    parser.add_argument('--num_ids', default=22, type=int, help='number of identities')
    parser.add_argument('--num_channel', default=5, type=int, help='number of temporal bins of event-voxel')
    parser.add_argument('--file_name', default='test_results', type=str, help='log file name')
    parser.add_argument('--mat', default='', type=str, help='name for saving representation')

    "IF train with denoise"
    parser.add_argument('--raw', default=0, type=int, help='if 0, use denoise')
    opt = parser.parse_args()

    """Save Log History"""
    sys.stdout = logging.Logger(os.path.join(opt.file_name+'/'+opt.name+'/', 'log.txt'))

    """Set GPU"""
    gpu_ids = []
    str_gpu_ids = opt.gpu_ids.split(',')
    for str_id in str_gpu_ids:
        gpu_ids.append(int(str_id))
    # torch.cuda.set_device(0)
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    cudnn.enabled = True#为固定的网络框架搜索合适的实现算法，加速网络
    cudnn.benchmark = True

    """re-id Model"""
    reid_model = EvReId(class_num=22, num_channel=opt.num_channel)
    reid_model = load_network(reid_model, opt.model_path)
    reid_model = reid_model.to(device) #.cuda()

    """Save Dir"""
    dir_name = os.path.join('./' + opt.file_name, opt.name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    # save opts
    with open('%s/opts.json'%dir_name, 'w') as fp:
        json.dump(vars(opt), fp, indent=1)

    """Start Test"""
    test_reid(reid_model)