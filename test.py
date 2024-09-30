from __future__ import print_function, division
import argparse
import json
import sys
import torch.backends.cudnn as cudnn
from reid_utils import RandomIdentitySampler, logging, RandomErasing
from reid_utils.test_utils import *
from models.AN_EvReId_model import EvReId
from data.dataloader import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from reid_utils import IterLoader
import numpy as np

def make_event_preview(events, mode='red-blue', num_bins_to_show=-1):
    # events: [1 x C x H x W] event tensor
    # mode: 'red-blue' or 'grayscale'
    # num_bins_to_show: number of bins of the voxel grid to show. -1 means show all bins.
    assert(mode in ['red-blue', 'grayscale'])
    if num_bins_to_show < 0:
        sum_events = torch.sum(events[0, :, :, :], dim=0)
    else:
        sum_events = torch.sum(events[0, -num_bins_to_show:, :, :], dim=0)

    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.ones((sum_events.shape[0], sum_events.shape[1], 3), dtype=np.uint8) * 100
        b = event_preview[:, :, 0]
        r = event_preview[:, :, 2]
        g = event_preview[:, :, 1]
        b[sum_events != 0] = 0
        r[sum_events != 0] = 0
        g[sum_events != 0] = 0
        b[sum_events > 0] = 100
        r[sum_events < 0] = 100

    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        sum_events = sum_events.numpy()
        m, M = -10.0, 10.0
        event_preview = np.clip((255.0 * (sum_events - m) / (M - m)).astype(np.uint8), 0, 255)

    return event_preview

def test_reid(model):
    model.eval()
    dataloaders_pure, dataset_pure = load_pure_data(opt)
    datasetnew = voxelDatasetcam_new(dataset_pure)
    new_loader = IterLoader(torch.utils.data.DataLoader(datasetnew,
                                                        batch_size=16, num_workers=8,
                                                        drop_last=True,
                                                        sampler=RandomIdentitySampler(datasetnew,
                                                                                      16),
                                                        shuffle=False, pin_memory=True),
                            length=200)
    with torch.no_grad():
        print('Start Inference...')  #每个行人都出现在所有cam
        features = []
        global_labels = []
        all_cams = []
        eventvisual = np.zeros((256, 384, 192, 3), dtype=np.uint8) # 8*instance
        new_loader.new_epoch()
        for i in range(8): # num id to show
            data = new_loader.next()
            input_voxel, labels, cams, acculabels, indexs = data
            _, embed_feats, _ = model(input_voxel.to(device))
            features.append(embed_feats.cpu())
            global_labels.append(labels)
            all_cams.append(cams)
            for j in range(len(labels)):
                eventpreview = make_event_preview(input_voxel[j:j+1, :, :, :])
                eventvisual[i * 16 + j, :, :, :] = eventpreview # instance
        features = torch.cat(features, dim=0).numpy()
        global_labels = torch.cat(global_labels, dim=0).numpy()
        all_cams = torch.cat(all_cams, dim=0).numpy()
    tsne = TSNE(n_components=2, init='pca', random_state=501, perplexity=5, n_iter=5000)
    X_tsne = tsne.fit_transform(features)
    X_norm = X_tsne
    #x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    #X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    fig, ax  = plt.subplots(figsize=(20, 20))
    ax.scatter(X_norm[:, 0], X_norm[:, 1])

    linestype = ['-', '--', '-.', ':']

    edgecolor = ['xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:brown', 'xkcd:brown', 'mlhi',
                 'm', 'b', 'xkcd:orange', 'g', 'xkcd:magenta', 'xkcd:yellow',
                 'xkcd:sky blue', 'xkcd:grey', 'c', 'xkcd:light purple', 'k', 'xkcd:dark green',
                 'y', 'xkcd:lavender', 'r', 'xkcd:tan']

    for i in range(X_norm.shape[0]):
        imagebox = OffsetImage(eventvisual[i], zoom=0.3)
        ab = AnnotationBbox(imagebox, (X_norm[i, 0], X_norm[i, 1]),
                            boxcoords="data",
                            frameon=True,
                            pad=0.0,
                            bboxprops=dict(edgecolor=edgecolor[global_labels[i]], facecolor='none',
                                           linestyle=linestype[all_cams[i]], linewidth=8,
                                           boxstyle='round,pad=0.0'))
        ax.add_artist(ab)
    plt.axis('off')
    plt.show()

    ''''
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(global_labels[i]), color=plt.cm.Set1(global_labels[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()
    '''
    return

def extract_feature(model, data):#?????,USLreid没有这种做法
    pool5_features = torch.FloatTensor()
    embed_features = torch.FloatTensor()
    count = 0
    device = next(model.parameters()).device
    with torch.no_grad():
            img, labels, cams = data
            n, c, h, w = img.size()
            count += n
            for i in range(2):  # ？？？？某种数据增强的方法，为何要在推理的时候
                if (i == 1):
                    img = fliplr(img)
                input_img = Variable(img.to(device))  # .cuda())
                _, embed_feature, pool5_feature = model(input_img)  # 256，2048
                if (i == 0):
                    ff_pool5 = torch.FloatTensor(n, pool5_feature.size(1)).zero_()
                    ff_embed = torch.FloatTensor(n, embed_feature.size(1)).zero_()
                f_pool5 = pool5_feature.data.cpu()
                ff_pool5 = ff_pool5 + f_pool5
                f_embed = embed_feature.data.cpu()
                ff_embed = ff_embed + f_embed
            fnorm_pool5 = torch.norm(ff_pool5, p=2, dim=1, keepdim=True)  # l2标准化
            fnorm_embed = torch.norm(ff_embed, p=2, dim=1, keepdim=True)
            ff_pool5 = ff_pool5.div(fnorm_pool5.expand_as(ff_pool5))
            ff_embed = ff_embed.div(fnorm_embed.expand_as(ff_embed))
            pool5_features = torch.cat((pool5_features, ff_pool5), 0)
            embed_features = torch.cat((embed_features, ff_embed), 0)
    return pool5_features, embed_features

def test_reid1(model):
    model.eval()
    image_datasets = voxelDataset11(mode='gallery', arg=opt)
    new_loader = IterLoader(torch.utils.data.DataLoader(image_datasets,
                                                        batch_size=16, num_workers=8,
                                                        drop_last=True,
                                                        sampler=RandomIdentitySampler(image_datasets,
                                                                                      16),
                                                        shuffle=False, pin_memory=True),
                            length=200)
    with torch.no_grad():
        print('Start Inference...')  #每个行人都出现在所有cam
        features = []
        global_labels = []
        all_cams = []
        new_loader.new_epoch()
        for i in range(11): # num id to show
            data = new_loader.next()
            input_voxel, labels, cams = data
            _, embed_feats = extract_feature(model, data)
            features.append(embed_feats.cpu())
            global_labels.append(labels)
            all_cams.append(cams)
        features = torch.cat(features, dim=0).numpy()
        global_labels = torch.cat(global_labels, dim=0).numpy()
        all_cams = torch.cat(all_cams, dim=0).numpy()
    tsne = TSNE(n_components=2, init='pca', random_state=501, perplexity=10, n_iter=5000)
    X_tsne = tsne.fit_transform(features)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(global_labels[i]), color=plt.cm.Set1(global_labels[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()

    return

def test_reid2(model):
    model.eval()
    image_datasets = voxelDataset11(mode='gallery', arg=opt)
    new_loader = IterLoader(torch.utils.data.DataLoader(image_datasets,
                                                        batch_size=16, num_workers=8,
                                                        drop_last=True,
                                                        sampler=RandomIdentitySampler(image_datasets,
                                                                                      16),
                                                        shuffle=False, pin_memory=True),
                            length=200)
    with torch.no_grad():
        print('Start Inference...')  #每个行人都出现在所有cam
        eventvisual = np.zeros((128, 384, 192, 3), dtype=np.uint8)  # 8*instance
        features = []
        global_labels = []
        all_cams = []
        new_loader.new_epoch()
        for i in range(8): # num id to show
            data = new_loader.next()
            input_voxel, labels, cams = data
            _, embed_feats = extract_feature(model, data)
            features.append(embed_feats.cpu())
            global_labels.append(labels)
            all_cams.append(cams)
            for j in range(len(labels)):
                eventpreview = make_event_preview(input_voxel[j:j+1, :, :, :])
                eventvisual[i * 16 + j, :, :, :] = eventpreview # instance
        features = torch.cat(features, dim=0).numpy()
        global_labels = torch.cat(global_labels, dim=0).numpy()
        all_cams = torch.cat(all_cams, dim=0).numpy()
    tsne = TSNE(n_components=2, init='pca', random_state=501, perplexity=10, n_iter=5000)
    X_tsne = tsne.fit_transform(features)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.scatter(X_norm[:, 0], X_norm[:, 1])

    linestype = ['-', '--', '-.', ':']

    edgecolor = ['m', 'b', 'g', 'c', 'k', 'y', 'r', 'xkcd:purple', 'xkcd:orange', 'xkcd:tan', 'xkcd:dark green']

    for i in range(X_norm.shape[0]):
        imagebox = OffsetImage(eventvisual[i], zoom=0.3)
        ab = AnnotationBbox(imagebox, (X_norm[i, 0], X_norm[i, 1]),
                            boxcoords="data",
                            frameon=True,
                            pad=0.0,
                            bboxprops=dict(edgecolor=edgecolor[global_labels[i]], facecolor='none',
                                           linestyle=linestype[all_cams[i]], linewidth=8,
                                           boxstyle='round,pad=0.0'))
        ax.add_artist(ab)
    plt.axis('off')
    plt.show()

    return

def load_pure_data(opt):
    if opt.represent == "voxel" and opt.raw == 1:
        image_datasets = voxelDatasetcam(mode='train', arg=opt)
        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
                                                      shuffle=False, num_workers=8, pin_memory=True)
        return dataloaders, image_datasets


def load_network(network, path):
    pretrained_dict = torch.load(path)
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    """Parser"""
    parser = argparse.ArgumentParser(description='Visaulization')
    parser.add_argument('--model_path', default='training3/net_best.pth', type=str,
                        help='path to pretrained Event-ReId model wieghts')
    parser.add_argument('--represent', default='voxel', type=str, help='representation of events for reid')
    parser.add_argument('--gpu_ids', default='0', type=str,help='gpu_ids: e.g. 0  0,1')
    parser.add_argument('--name', default='Denoise_Event_ReId', type=str, help='output model name')
    parser.add_argument('--num_ids', default=22, type=int, help='number of identities')
    parser.add_argument('--num_channel', default=5, type=int, help='number of temporal bins of event-voxel')
    parser.add_argument('--file_name', default='test_results', type=str, help='log file name')
    parser.add_argument('--mat', default='', type=str, help='name for saving representation')
    parser.add_argument('--batchsize', default=16, type=int, help='batchsize')

    "IF train with denoise"
    parser.add_argument('--raw', default=1, type=int, help='if 0, use denoise')
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
    device  = torch.device('cuda' if use_gpu else 'cpu')
    cudnn.enabled = True#为固定的网络框架搜索合适的实现算法，加速网络
    cudnn.benchmark = True

    "确保可复现"
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)  # 为CPU设置随机种子
    torch.cuda.manual_seed(123)  # 为当前GPU设置随机种子

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
    test_reid1(reid_model)
