from __future__ import print_function, division
import argparse
import json
import time
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from reid_utils import RandomIdentitySampler, logging, RandomErasing
from reid_utils.CamAwareMemory import CAPMemory, CAPMemorymy3
from reid_utils.test_utils import *
from models.AN_EvReId_model import EvReId, EvReIdcam
from data.dataloader import *
from reid_losses import TripletLoss1
from torch.utils.tensorboard import SummaryWriter
from e2vid_utils.e2vid_image_utils import *
from e2vid_utils.utils.loading_utils import load_E2VID
from reid_utils import IterLoader
from evaluator import evaluator, evaluatorcam2
from reid_utils.sampler import ClassUniformlySampler
from bisect import bisect_right
from torch import nn


def id_proxy_memory(E2VID, model, loader, device):
    model.eval()
    with torch.no_grad():
        print('Start Inference...')  #每个行人都出现在所有cam
        features = []
        global_labels = []
        all_cams = []
        for (image_grids, labels, cams, indexs) in loader:
            img_gray, stat = E2VID(image_grids.to(device), None)
            img_input = torch.cat([image_grids.to(device), img_gray], dim=1)
            _, _, embed_feats, _ = model(img_input)
            features.append(embed_feats.cpu())
            global_labels.append(labels)
            all_cams.append(cams)
        features = torch.cat(features, dim=0).numpy()
        global_labels = torch.cat(global_labels, dim=0).numpy()

        unique_label = np.unique(global_labels)

        print('  features: shape= {}'.format(features.shape))

        print('re-computing initialized intra-ID feature...')
        intra_id_features = np.zeros((len(unique_label), features.shape[1]), dtype=np.float32)

        for lbl in np.unique(unique_label):
            if lbl >= 0:
                ind = np.where(global_labels == lbl)[0]
                id_feat = np.mean(features[ind], axis=0)  # class proxy
                intra_id_features[lbl, :] = id_feat
        intra_id_features = intra_id_features / np.linalg.norm(intra_id_features, axis=1, keepdims=True)  # L2
        intra_id_features = torch.tensor(intra_id_features)
        unique_label = torch.tensor(unique_label)
        return unique_label, intra_id_features

def train_model(model_ReId, E2VID, optimizer, scheduler, device, num_epochs=50):  #PSA+ISA+HML+ID +Concat
    E2VID = E2VID.eval()
    start_time = time.time()
    writer = SummaryWriter()
    dataloaders_pure, dataset_pure = load_pure_data(opt)

    "construct camera proxy memory bank"
    intra_id_labels, intra_id_features = id_proxy_memory(E2VID, model_ReId, dataloaders_pure, device)
    cap_memory = CAPMemorymy3(beta=opt.inv_beta, alpha=opt.inv_alpha, all_img_cams=dataset_pure.cam_list, global_labels=dataset_pure.label_list,
                           proxy_memory=intra_id_features, proxy_id=intra_id_labels).to(device)

    "construct new loader with ClassUniformlySampler"
    if opt.raw == 1:
        datasetnew = voxelDatasetcam_new2(dataset_pure)

    new_loader = IterLoader(torch.utils.data.DataLoader(datasetnew,
                                                        batch_size=opt.batchsize, num_workers=opt.num_workers,
                                                        drop_last=True,
                                                        sampler=RandomIdentitySampler(datasetnew,
                                                                                      opt.num_instances),
                                                        shuffle=False, pin_memory=True),
                            length=200)

    dataloaders_test, image_datasets_test = load_test_data(opt)
    bestmap = 0.0
    bestepoch = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train']:
            if epoch >= 1:
                cap_memory.init_memory(E2VID, model_ReId, dataloaders_pure, device)  #每一轮都初始化memory
            model_ReId.train(True)  #每次验证后再调到训练模式

            adjust_lr_softmax(optimizer, epoch, warmup_ep=10, warmup_factor=0.05)

            new_loader.new_epoch()

            running_losssoft = 0.0
            running_losstri = 0.0
            running_losscl = 0.0
            running_correct_category = 0

            for i in range(200):
                data = new_loader.next()
                input_voxel, labels, cams, acculabels, indexs = data
                input_voxel = Variable(input_voxel.to(device))
                labels = Variable(labels.to(device))
                cams = Variable(cams.to(device))
                acculabels = Variable(acculabels.to(device))
                indexs = Variable(indexs.to(device))

                optimizer.zero_grad()

                """ReId model training"""
                img_gray, stat = E2VID(input_voxel, None)
                img_gray_detach = img_gray.detach()
                img_gray_input = torch.cat([input_voxel, img_gray_detach], dim=1)
                [category, feature, feature_l2, _] = model_ReId(img_gray_input)  # feature:embed_feature

                """Event ReId Losses"""
                _, category_preds = torch.max(category.data, 1)

                loss_softmax = criterion_softmax(category, labels)
                loss_triplet, _, _ = criterion_triplet(feature, labels)
                # loss_triplet = torch.tensor([0.]).to(device)
                loss_cl = cap_memory(feature_l2, labels, acculabels, cams, epoch=epoch, batch_ind=i)

                loss_reid = loss_softmax + loss_cl * 0.7 + loss_triplet

                # writer.add_scalars(f'batch_loss', {'reid_loss': loss_reid}, batch_count)
                print('batch loss', loss_softmax.item(), loss_triplet.item(), loss_cl.item())

                """Total Loss"""
                loss = loss_reid
                loss.backward()
                optimizer.step()

                # statistics
                running_losssoft += loss_softmax.item()
                running_losstri += loss_triplet.item()
                running_losscl += loss_cl.item()

                category_preds = category.data.max(1)[1]
                running_correct_category += torch.sum(category_preds == labels.data)

            epoch_losssoft = running_losssoft / 200
            epoch_losstri = running_losstri / 200
            epoch_losscl = running_losscl / 200
            epoch_acc = running_correct_category.cpu().numpy() / 200 / opt.batchsize
            print('{} Losssoft: {:.4f} Losstri: {:.4f} Losscl: {:.4f} Acc_category: {:.4f}'.format(phase, epoch_losssoft, epoch_losstri, epoch_losscl, epoch_acc))

            mapp, map_embed = evaluatorcam2(opt, E2VID, model_ReId, dataloaders=dataloaders_test,
                                        image_datasets=image_datasets_test)
            is_best = (map_embed > bestmap)
            bestmap = max(map_embed, bestmap)
            if is_best:
                bestepoch = epoch
                save_network(model_ReId, 'best')

            print('{} evaluate_map: {:.4f} evaluate_mapembed: {:.4f}'.format(phase, mapp, map_embed))

            """tensorboard"""
            # writer.add_graph(model)
            writer.add_scalars(main_tag='training',
                               tag_scalar_dict={'losssoft': epoch_losssoft, 'losstri': epoch_losstri,
                                                'losscl': epoch_losscl, 'accuracy': epoch_acc, 'map': mapp,
                                                'mapembed': map_embed}, global_step=epoch)
    print("bestmap: {:.4f}  bestepoch: {}".format(bestmap, bestepoch))
    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 30, time_elapsed % 30))
    writer.close()
    return model_ReId


def save_network(network1, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path1 = os.path.join('./' + opt.file_name, save_filename)
    torch.save(network1.cpu().state_dict(), save_path1)
    if torch.cuda.is_available():
        network1.cuda(0)


def load_test_data(opt):
    if opt.represent == "voxel" and opt.raw == 1:
        image_datasets = {x: voxelDataset2(mode=x, arg=opt) for x in ['gallery', 'query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                      shuffle=False, num_workers=8) for x in ['gallery', 'query']}
        return dataloaders, image_datasets


def adjust_lr_softmax(optimizer, ep, warmup_ep, warmup_factor):
    alpha = float(ep) / float(warmup_ep)
    warmup_factor = warmup_factor * (1 - alpha) + alpha
    if ep < warmup_ep:
        lr = 0.01 * warmup_factor
    elif ep < 40:
        lr = 0.01
    elif ep < 60:
        lr = 0.001
    elif ep < 80:
        lr = 0.0005
    else:
        lr = 0.0001
    for index in range(len(optimizer.param_groups)):
        if index == 0:
            optimizer.param_groups[index]['lr'] = lr * 0.1
        else:
            optimizer.param_groups[index]['lr'] = lr

def load_pure_data(opt):
    if opt.represent == "voxel" and opt.raw == 1:
        image_datasets = voxelDatasetcam2(mode='train', arg=opt)
        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
                                                      shuffle=False, num_workers=8, pin_memory=True)
        return dataloaders, image_datasets


def set_optimizer(model):
    ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': model.model.fc.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=0.001, weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
    return optimizer


if __name__ == '__main__':#0.7

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    """Parser"""
    parser = argparse.ArgumentParser(description='Event_ReId_cam')
    parser.add_argument('--represent', default='voxel', type=str, help='representation of events for reid')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1')
    parser.add_argument('--name', default='Event_ReId_cam', type=str, help='output model name')
    parser.add_argument('--num_ids', default=22, type=int, help='number of identities')
    parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
    parser.add_argument('--ReId_loss', default='softmax+intra cl+inter cl', type=str, help='choice of reid loss')
    parser.add_argument('--num_instances', default=8, type=int, help='per cam per id instances')
    parser.add_argument('--num_Bin', default=6, type=int, help='number of channels of spatiotemporal event-voxel')
    parser.add_argument('--epoch', default=60, type=int, help='training epoch')
    parser.add_argument('--file_name', default='training3', type=str, help='file name to save weights and log file')
    parser.add_argument('--iter', default=200, type=int, help='iteration of each batch')
    parser.add_argument('--num_workers', default=8, type=int, help='num workers for dataloader')
    parser.add_argument('--margin', default=10, type=float, help='triplet loss margin')

    "IF train with denoise"
    parser.add_argument('--raw', default=1, type=int, help='if 0, use denoise')

    "optimizer"
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    "e2vid"
    parser.add_argument('--e2vid_path', default='e2vid_utils/pretrained/E2VID_lightweight.pth.tar', type=str,
                        help='path to e2vid weights')

    "cl loss para"
    parser.add_argument('--inv_alpha', type=float, default=0.99, help='update rate for the memory')
    parser.add_argument('--inv_beta', type=float, default=0.07, help='temperature for contrastive loss')

    opt = parser.parse_args()

    """Save Log History"""
    sys.stdout = logging.Logger(os.path.join(opt.file_name + '/' + opt.name + '/', 'log.txt'))

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
    reid_model = EvReIdcam(class_num=opt.num_ids, num_channel=opt.num_Bin)

    """Optimizer"""
    optimizer = set_optimizer(reid_model)
    reid_model = reid_model.to(device)

    """Event-to-Video Model"""
    E2VID = load_E2VID(opt.e2vid_path, device)
    E2VID = E2VID.to(device)

    """Set ReId Loss function"""
    criterion_softmax = nn.CrossEntropyLoss()  # CE就是softmax，要避免出现log0
    criterion_triplet = TripletLoss1(opt.margin)

    """Save Dir"""
    dir_name = os.path.join('./' + opt.file_name, opt.name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    # save opts
    with open('%s/opts.json' % dir_name, 'w') as fp:
        json.dump(vars(opt), fp, indent=1)

    """Start Training"""
    model_ = train_model(reid_model, E2VID, optimizer, scheduler=None, device=device, num_epochs=opt.epoch)