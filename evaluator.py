from __future__ import print_function, division
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
from data.dataloader import *

def evaluator(arg, model, dataloaders, image_datasets):
    model.eval()
    print('-' * 10)
    print('test model now...')

    gallery_path = image_datasets['gallery'].name_list
    query_path = image_datasets['query'].name_list
    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    gallery_feature, gallery_feature_embed = extract_feature(model, dataloaders['gallery'])
    query_feature, query_feature_embed = extract_feature(model, dataloaders['query'])

    CMC = torch.IntTensor(len(gallery_label)).zero_()

    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i],
                                   gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    mapp = ap / len(query_label)

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('Pool5-Feature top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature_embed[i], query_label[i], query_cam[i],
                                   gallery_feature_embed, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    map_embed = ap / len(query_label)

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('Embed-Feature top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

    return mapp, map_embed

def evaluatorcam(arg, model, dataloaders, image_datasets):
    model.eval()
    print('-' * 10)
    print('test model now...')

    gallery_path = image_datasets['gallery'].name_list
    query_path = image_datasets['query'].name_list
    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    gallery_feature, gallery_feature_embed = extract_featurecam(model, dataloaders['gallery'])
    query_feature, query_feature_embed = extract_featurecam(model, dataloaders['query'])

    CMC = torch.IntTensor(len(gallery_label)).zero_()

    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i],
                                   gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    mapp = ap / len(query_label)

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('Pool5-Feature top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature_embed[i], query_label[i], query_cam[i],
                                   gallery_feature_embed, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    map_embed = ap / len(query_label)

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('Embed-Feature top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

    return mapp, map_embed

def evaluatorcam2(arg, E2VID, model, dataloaders, image_datasets):
    print('-' * 10)
    print('test model now...')

    gallery_path = image_datasets['gallery'].name_list
    query_path = image_datasets['query'].name_list
    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    gallery_feature, gallery_feature_embed = extract_featurecam3(E2VID, model, dataloaders['gallery'])
    query_feature, query_feature_embed = extract_featurecam3(E2VID, model, dataloaders['query'])

    CMC = torch.IntTensor(len(gallery_label)).zero_()

    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i],
                                   gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    mapp = ap / len(query_label)

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('Pool5-Feature top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature_embed[i], query_label[i], query_cam[i],
                                   gallery_feature_embed, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    map_embed = ap / len(query_label)

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('Embed-Feature top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

    return mapp, map_embed

def evaluatorcam3(arg, E2VID, model, dataloaders, image_datasets):
    print('-' * 10)
    print('test model now...')

    gallery_path = image_datasets['gallery'].name_list
    query_path = image_datasets['query'].name_list
    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    gallery_feature, gallery_feature_embed = extract_featurecam2(E2VID, model, dataloaders['gallery'])
    query_feature, query_feature_embed = extract_featurecam2(E2VID, model, dataloaders['query'])

    CMC = torch.IntTensor(len(gallery_label)).zero_()

    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i],
                                   gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    mapp = ap / len(query_label)

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('Pool5-Feature top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature_embed[i], query_label[i], query_cam[i],
                                   gallery_feature_embed, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    map_embed = ap / len(query_label)

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('Embed-Feature top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

    return mapp, map_embed