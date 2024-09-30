from collections import defaultdict

import numpy as np
import random
import torch
from torch.utils.data import Dataset
#from torchvision import transforms
from PIL import Image
import os
from scipy.special import expit
from torch.nn import functional as F
import cv2



def id_cam(file_paths, mode = 'train'):
    """extract ID and camera Number"""
    id = []
    cam = []
    train_ids = [1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
    test_ids = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32]
    for file in file_paths:
        id_ = int(file.split("_")[0])
        if mode == 'train':
            id_ = train_ids.index(id_)
        else:
            id_ = test_ids.index(id_)
        cam_ = file.split("_")[1]
        id.append(id_)
        cam.append(cam_)
    return id, cam

def id_camlabel(file_paths, mode = 'train'):
    """extract ID and camera Number"""
    id = []
    cam = []
    train_ids = [1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33]
    test_ids = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32]
    cams = ['c1', 'c2', 'c3', 'c4']
    for file in file_paths:
        id_ = int(file.split("_")[0])
        if mode == 'train':
            id_ = train_ids.index(id_)
        else:
            id_ = test_ids.index(id_)
        cam_ = file.split("_")[1]
        cam_ = cams.index(cam_)
        id.append(id_)
        cam.append(cam_)
    return id, cam

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def extract_bbox(event):
    """Image"""
    xi = event[:, 1].astype(int)
    yi = event[:, 2].astype(int)
    event_frame = np.zeros((480, 640, 3), np.uint8)

    """replace event coordinates with non-zero value"""
    event_frame[yi, xi, :] = [255, 255, 255]

    """convert np.array to image"""
    event_frame = Image.fromarray(event_frame)

    """extract bbox"""
    x1, y1, x2, y2 = event_frame.getbbox()

    """Bbox Padding"""
    if x1 > 4:
        x1 = x1-2
    if y1 > 4:
        y1 = y1-2
    if y2 < 476:
        y2 = y2 + 2
    if x2 < 636:
        x2 = x2 + 2
    bbox = x1, y1, x2, y2

    return bbox


def normalize_voxel(events):  # 标准化
    nonzero_ev = (events != 0)
    num_nonzeros = nonzero_ev.sum()

    if num_nonzeros > 0:
        """ compute mean and stddev of the **nonzero** elements of the event tensor
        we do not use PyTorch's default mean() and std() functions since it's faster
        to compute it by hand than applying those funcs to a masked array """

        mean = events.sum() / num_nonzeros
        stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)#连0元素也算进去了
        mask = nonzero_ev.float()
        events = mask * (events - mean) / stddev

    return events


def events_to_voxel_grid(events, num_bins, width, height, is_hard=False):
    """
    Build a voxel grid with bilinear interpolation from a set of events in the time domain.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """
    width = int(width)
    height = int(height)
    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float64).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int)
    ys = events[:, 2].astype(np.int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid

def spatial_temporal_filter2(grid): #5通道分别进行连通图滤波，在进行huo运算
    mask_over = np.zeros((grid[0].shape[0], grid[0].shape[1]), np.integer)
    for i in range(len(grid)):
        binary_image = np.zeros((grid[i].shape[0], grid[i].shape[1]), np.uint8)
        binary_image[grid[i] != 0] = 255
        #binary_image = binary_image.astype(np.uint8)
        mask = np.zeros((grid[i].shape[0], grid[i].shape[1]), np.integer)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        for j in range(1, num_labels):
            index = (labels == j)  # 这一步是通过labels确定区域位置，让labels信息赋给mask数组，再用mask数组做img数组的索引
            if stats[j][4] > 1:  # 300是像素个数 可以随便调
                mask[index] = 1
                # 面积大于300的区域涂白留下，小于300的涂0抹去
            else:
                mask[index] = 0
        mask_over = mask_over | mask
    for i in range(len(grid)):
        grid[i] = grid[i] * mask_over
    return grid


class voxelDataset2(Dataset):

    def __init__(self, event_dir="data/", mode='train', transform=None, arg=None):
        self.mode = mode
        self.transform = transform
        self.event_dir = event_dir
        self.arg = arg
        supported_modes = ('train', 'query', 'gallery')
        assert self.mode in supported_modes, print("Only support mode from {}".format(supported_modes))
        if self.mode == 'train':
            self.name_list = [file for file in os.listdir(event_dir + self.mode + '/') if
                              file.endswith('.txt') and os.path.getsize(event_dir + self.mode + '/' + file)]
            self.name_list = sorted(self.name_list)
            self.label_list, self.cam_list = id_camlabel(self.name_list, mode=self.mode)
        elif self.mode == 'query':
            name_list_gallery = [file for file in os.listdir(event_dir + 'test' + '/') if
                                 file.endswith('.txt') and os.path.getsize(event_dir + 'test' + '/' + file)]
            name_list_gallery = sorted(name_list_gallery)
            label_list_gallery, cam_list_gallery = id_camlabel(name_list_gallery, mode=self.mode)
            self.name_list = []
            self.label_list = []
            self.cam_list = []
            self.query_index = []
            test_indexs = np.arange(start=0, stop=len(name_list_gallery), step=1, dtype=int)
            for i in range(11):
                for j in [0, 1, 2, 3]:
                    indexs_label = [index for index, value in enumerate(label_list_gallery) if value == i]
                    indexs_cam = [index for index, value in enumerate(cam_list_gallery) if value == j]
                    indexs = [index for index in indexs_label if index in indexs_cam]
                    index = random.sample(indexs, k=1)
                    self.query_index.append(test_indexs[index[0]])
                    self.name_list.append(name_list_gallery[index[0]])
                    self.label_list.append(label_list_gallery[index[0]])
                    self.cam_list.append(cam_list_gallery[index[0]])
        elif self.mode == 'gallery':
            name_list_pure = [file for file in os.listdir(event_dir + 'test' + '/') if
                              file.endswith('.txt') and os.path.getsize(event_dir + 'test' + '/' + file)]
            name_list_pure = sorted(name_list_pure)
            name_list_query = []
            query_index = []
            test_indexs = np.arange(start=0, stop=len(name_list_pure), step=1, dtype=int)
            self.name_list = [x for x in name_list_pure if x not in name_list_query]
            self.gallery_index = [x for x in test_indexs if x not in query_index]
            self.label_list, self.cam_list = id_camlabel(self.name_list, mode=self.mode)

    def __getitem__(self, index):  # 一个event chunk一个voxel grid?
        if self.mode == 'train':
            events = np.loadtxt(self.event_dir + self.mode + '/' + self.name_list[index], dtype=np.float64,
                                delimiter=' ',
                                usecols=(0, 1, 2, 3))
        elif self.mode == 'gallery':
            events = np.loadtxt(self.event_dir + 'test' + '/' + self.name_list[index], dtype=np.float64,
                                delimiter=' ',
                                usecols=(0, 1, 2, 3))
        elif self.mode == 'query':
            events = np.loadtxt(self.event_dir + 'test' + '/' + self.name_list[index], dtype=np.float64,
                                delimiter=' ',
                                usecols=(0, 1, 2, 3))

        """ 640x480 => bbox(x_, y_), as we have already clean/removed the events outside bbox"""
        events[:, 1] = events[:, 1] - min(events[:, 1])  # x - x_min
        events[:, 2] = events[:, 2] - min(events[:, 2])  # y - y_min
        events[:, 0] = (events[:, 0] - min(events[:, 0])) * 1000  # t - t_min

        """event => voxel-grid"""
        voxel_grid = events_to_voxel_grid(events, num_bins=5, width=max(events[:, 1]) + 1, height=max(events[:, 2]) + 1)

        """Numpy to Tensor"""
        voxel_grid = torch.tensor(voxel_grid)

        """Numpy to Tensor"""
        voxel_grid = normalize_voxel(voxel_grid)

        """Resize Tensor"""
        voxel_grid = F.interpolate(voxel_grid.unsqueeze(0).float(), size=(384, 192), mode='nearest').squeeze(0)

        label = self.label_list[index]
        cam = self.cam_list[index]

        return voxel_grid, label

    def __len__(self):
        return len(self.name_list)

    def get_index_dict(self):
        index_dict = defaultdict(list)
        index = 0
        for i in self.label_list:
            index_dict[i].append(index)
            index += 1
        return index_dict


class voxelDataset(Dataset):
    
    def __init__(self, event_dir="data/", mode='train', transform=None, arg=None):
        self.mode = mode
        self.transform = transform
        self.event_dir = event_dir
        self.arg = arg
        supported_modes = ('train', 'query', 'gallery')
        assert self.mode in supported_modes, print("Only support mode from {}".format(supported_modes))
        if self.mode == 'train':
            self.name_list = [file for file in os.listdir(event_dir + self.mode + '/') if file.endswith('.txt') and os.path.getsize(event_dir + self.mode + '/' + file)]
            self.name_list = sorted(self.name_list)
            self.label_list, self.cam_list = id_camlabel(self.name_list, mode=self.mode)
        elif self.mode == 'query':
            name_list_gallery = [file for file in os.listdir(event_dir + 'test' + '/') if file.endswith('.txt') and os.path.getsize(event_dir + 'test' + '/' + file)]
            name_list_gallery = sorted(name_list_gallery)
            label_list_gallery, cam_list_gallery = id_camlabel(name_list_gallery, mode=self.mode)
            self.name_list = []
            self.label_list = []
            self.cam_list = []
            self.query_index = []
            test_indexs = np.arange(start=0, stop=len(name_list_gallery), step=1, dtype=int)
            for i in range(11):
                for j in [0, 1, 2, 3]:
                    indexs_label = [index for index, value in enumerate(label_list_gallery) if value == i]
                    indexs_cam = [index for index, value in enumerate(cam_list_gallery) if value == j]
                    indexs = [index for index in indexs_label if index in indexs_cam]
                    index = random.sample(indexs, k=1)
                    self.query_index.append(test_indexs[index[0]])
                    self.name_list.append(name_list_gallery[index[0]])
                    self.label_list.append(label_list_gallery[index[0]])
                    self.cam_list.append(cam_list_gallery[index[0]])
        elif self.mode == 'gallery':
            name_list_pure = [file for file in os.listdir(event_dir + 'test' + '/') if file.endswith('.txt') and os.path.getsize(event_dir + 'test' + '/' + file)]
            name_list_pure = sorted(name_list_pure)
            name_list_query = []
            query_index = []
            test_indexs = np.arange(start=0, stop=len(name_list_pure), step=1, dtype=int)
            self.name_list = [x for x in name_list_pure if x not in name_list_query]
            self.gallery_index = [x for x in test_indexs if x not in query_index]
            self.label_list, self.cam_list = id_camlabel(self.name_list, mode=self.mode)

    def __getitem__(self, index):#一个event chunk一个voxel grid?
        if self.mode == 'train':
            events = np.loadtxt(self.event_dir + self.mode + '/' + self.name_list[index], dtype=np.float64,
                                delimiter=' ',
                                usecols=(0, 1, 2, 3))
        elif self.mode == 'gallery':
            events = np.loadtxt(self.event_dir + 'test' + '/' + self.name_list[index], dtype=np.float64,
                                delimiter=' ',
                                usecols=(0, 1, 2, 3))
        elif self.mode == 'query':
            events = np.loadtxt(self.event_dir + 'test' + '/' + self.name_list[index], dtype=np.float64,
                                delimiter=' ',
                                usecols=(0, 1, 2, 3))

        is_hard = (len(events) >= 1000)  # 1是正常样本
        """ 640x480 => bbox(x_, y_), as we have already clean/removed the events outside bbox"""
        events[:, 1] = events[:, 1] - min(events[:, 1])  # x - x_min
        events[:, 2] = events[:, 2] - min(events[:, 2])  # y - y_min
        events[:, 0] = (events[:, 0] - min(events[:, 0])) * 1000  # t - t_min

        """event => voxel-grid"""
        voxel_grid = events_to_voxel_grid(events, num_bins=5, width=max(events[:, 1]) + 1, height=max(events[:, 2]) + 1,
                                        is_hard=is_hard)

        """Numpy to Tensor"""
        voxel_grid = torch.tensor(voxel_grid)

        """Numpy to Tensor"""
        voxel_grid = normalize_voxel(voxel_grid)

        """Resize Tensor"""
        voxel_grid = F.interpolate(voxel_grid.unsqueeze(0).float(), size=(384, 192), mode='nearest').squeeze(0)#每一次插值都是固定的吗

        label = self.label_list[index]
        cam = self.cam_list[index]

        return voxel_grid, label

    def __len__(self):
        return len(self.name_list)

    def get_index_dict(self):
        index_dict = defaultdict(list)
        index = 0
        for i in self.label_list:
            index_dict[i].append(index)
            index += 1
        return index_dict

class voxelDataset11(Dataset):

    def __init__(self, event_dir="data/", mode='train', transform=None, arg=None):
        self.mode = mode
        self.transform = transform
        self.event_dir = event_dir
        self.arg = arg
        supported_modes = ('train', 'query', 'gallery')
        assert self.mode in supported_modes, print("Only support mode from {}".format(supported_modes))
        if self.mode == 'gallery':
            name_list_pure = [file for file in os.listdir(event_dir + 'test1' + '/') if
                              file.endswith('.txt') and os.path.getsize(event_dir + 'test' + '/' + file) and
                              np.size(np.loadtxt(event_dir + 'test' + '/' + file,
                                      dtype=np.float64,
                                      delimiter=' ', usecols=(0, 1, 2, 3)), 0) >= 0]#3000
            name_list_pure = sorted(name_list_pure)
            name_list_query = []
            query_index = []
            test_indexs = np.arange(start=0, stop=len(name_list_pure), step=1, dtype=int)
            self.name_list = [x for x in name_list_pure if x not in name_list_query]
            self.gallery_index = [x for x in test_indexs if x not in query_index]
            self.label_list, self.cam_list = id_camlabel(self.name_list, mode=self.mode)

    def __getitem__(self, index):  # 一个event chunk一个voxel grid?
        if self.mode == 'gallery':
            events = np.loadtxt(self.event_dir + 'test' + '/' + self.name_list[index], dtype=np.float64,
                                delimiter=' ',
                                usecols=(0, 1, 2, 3))
        """ 640x480 => bbox(x_, y_), as we have already clean/removed the events outside bbox"""
        events[:, 1] = events[:, 1] - min(events[:, 1])  # x - x_min
        events[:, 2] = events[:, 2] - min(events[:, 2])  # y - y_min
        events[:, 0] = (events[:, 0] - min(events[:, 0])) * 1000  # t - t_min

        """event => voxel-grid"""
        voxel_grid = events_to_voxel_grid(events, num_bins=5, width=max(events[:, 1]) + 1, height=max(events[:, 2]) + 1)

        """Numpy to Tensor"""
        voxel_grid = torch.tensor(voxel_grid)

        """Numpy to Tensor"""
        voxel_grid = normalize_voxel(voxel_grid)

        """Resize Tensor"""
        voxel_grid = F.interpolate(voxel_grid.unsqueeze(0).float(), size=(384, 192), mode='nearest').squeeze(
            0)  # 每一次插值都是固定的吗

        label = self.label_list[index]
        cam = self.cam_list[index]

        return voxel_grid, label, cam

    def __len__(self):
        return len(self.name_list)

    def get_index_dict(self):
        index_dict = defaultdict(list)
        index = 0
        for i in self.label_list:
            index_dict[i].append(index)
            index += 1
        return index_dict


class voxelDatasetcam(Dataset):

    def __init__(self, event_dir="data/", mode='train', transform=None, arg=None):
        self.mode = mode
        self.transform = transform
        self.event_dir = event_dir
        self.arg = arg
        supported_modes = ('train', 'query', 'gallery')
        assert self.mode in supported_modes, print("Only support mode from {}".format(supported_modes))
        if self.mode == 'train':
            self.name_list = [file for file in os.listdir(event_dir + self.mode + '/') if
                              file.endswith('.txt') and os.path.getsize(event_dir + self.mode + '/' + file)]
            self.name_list = sorted(self.name_list)
            self.label_list, self.cam_list = id_camlabel(self.name_list, mode=self.mode)

    def __getitem__(self, index):  # 一个event chunk一个voxel grid?
        if self.mode == 'train':
            events = np.loadtxt(self.event_dir + self.mode + '/' + self.name_list[index], dtype=np.float64,
                                delimiter=' ',
                                usecols=(0, 1, 2, 3))
        is_hard = (len(events) >= 1000) #1是正常样本
        """ 640x480 => bbox(x_, y_), as we have already clean/removed the events outside bbox"""
        events[:, 1] = events[:, 1] - min(events[:, 1])  # x - x_min
        events[:, 2] = events[:, 2] - min(events[:, 2])  # y - y_min
        events[:, 0] = (events[:, 0] - min(events[:, 0])) * 1000  # t - t_min

        """event => voxel-grid"""
        voxel_grid = events_to_voxel_grid(events, num_bins=5, width=max(events[:, 1]) + 1, height=max(events[:, 2]) + 1, is_hard = is_hard)

        """Numpy to Tensor"""
        voxel_grid = torch.tensor(voxel_grid)

        """Numpy to Tensor"""
        voxel_grid = normalize_voxel(voxel_grid)

        """Resize Tensor"""
        voxel_grid = F.interpolate(voxel_grid.unsqueeze(0).float(), size=(384, 192), mode='nearest').squeeze(0)

        label = self.label_list[index]
        cam = self.cam_list[index]

        return voxel_grid, label, cam, index, is_hard

    def __len__(self):
        return len(self.name_list)

    def get_index_dict(self):
        index_dict = defaultdict(list)
        index = 0
        for i in self.label_list:
            index_dict[i].append(index)
            index += 1
        return index_dict

class voxelDatasetcam2(Dataset):

    def __init__(self, event_dir="data/", mode='train', transform=None, arg=None):
        self.mode = mode
        self.transform = transform
        self.event_dir = event_dir
        self.arg = arg
        supported_modes = ('train', 'query', 'gallery')
        assert self.mode in supported_modes, print("Only support mode from {}".format(supported_modes))
        if self.mode == 'train':
            self.name_list = [file for file in os.listdir(event_dir + self.mode + '/') if
                              file.endswith('.txt') and os.path.getsize(event_dir + self.mode + '/' + file)]
            self.name_list = sorted(self.name_list)
            self.label_list, self.cam_list = id_camlabel(self.name_list, mode=self.mode)

    def __getitem__(self, index):  # 一个event chunk一个voxel grid?
        if self.mode == 'train':
            events = np.loadtxt(self.event_dir + self.mode + '/' + self.name_list[index], dtype=np.float64,
                                delimiter=' ',
                                usecols=(0, 1, 2, 3))
        """ 640x480 => bbox(x_, y_), as we have already clean/removed the events outside bbox"""
        events[:, 1] = events[:, 1] - min(events[:, 1])  # x - x_min
        events[:, 2] = events[:, 2] - min(events[:, 2])  # y - y_min
        events[:, 0] = (events[:, 0] - min(events[:, 0])) * 1000  # t - t_min

        """event => voxel-grid"""
        voxel_grid = events_to_voxel_grid(events, num_bins=5, width=max(events[:, 1]) + 1, height=max(events[:, 2]) + 1)

        """Numpy to Tensor"""
        voxel_grid = (torch.tensor(voxel_grid))

        """Numpy to Tensor"""
        voxel_grid = normalize_voxel(voxel_grid)

        """Resize Tensor"""
        voxel_grid = F.interpolate(voxel_grid.unsqueeze(0).float(), size=(384, 192), mode='nearest').squeeze(0)

        label = self.label_list[index]
        cam = self.cam_list[index]

        return voxel_grid, label, cam, index

    def __len__(self):
        return len(self.name_list)

    def get_index_dict(self):
        index_dict = defaultdict(list)
        index = 0
        for i in self.label_list:
            index_dict[i].append(index)
            index += 1
        return index_dict

class voxelDatasetcam_new(Dataset):

    def __init__(self, pure_dataset=None, event_dir="data/", mode='train'):
        self.name_list = pure_dataset.name_list
        self.cam_list = pure_dataset.cam_list
        self.label_list = pure_dataset.label_list
        self.event_dir = event_dir
        self.mode = mode
        self.relabel_under_eachcam()

    def relabel_under_eachcam(self):
        all_label = np.array(self.label_list)
        all_cams = np.array(self.cam_list)
        self.accumulate_labels = np.zeros(all_label.shape, all_label.dtype)
        prev_id_count = 0
        self.id_count_each_cam = []
        for this_cam in np.unique(all_cams):
            percam_labels = all_label[all_cams == this_cam]
            unique_id = np.unique(percam_labels)
            self.id_count_each_cam.append(len(unique_id))
            id_dict = {ID: i for i, ID in enumerate(unique_id.tolist())}
            for i in range(len(percam_labels)):
                percam_labels[i] = id_dict[percam_labels[i]]  # relabel
            self.accumulate_labels[all_cams == this_cam] = percam_labels + prev_id_count
            prev_id_count += len(unique_id)
        print('  sum(id_count_each_cam)= {}'.format(sum(self.id_count_each_cam)))

    def __getitem__(self, index):  # 一个event chunk一个voxel grid?
        events = np.loadtxt(self.event_dir + self.mode + '/' + self.name_list[index], dtype=np.float64,
                            delimiter=' ',
                            usecols=(0, 1, 2, 3))

        is_hard = (len(events) >= 1000)  # 1是正常样本
        """ 640x480 => bbox(x_, y_), as we have already clean/removed the events outside bbox"""
        events[:, 1] = events[:, 1] - min(events[:, 1])  # x - x_min
        events[:, 2] = events[:, 2] - min(events[:, 2])  # y - y_min
        events[:, 0] = (events[:, 0] - min(events[:, 0])) * 1000  # t - t_min

        """event => voxel-grid"""
        voxel_grid = events_to_voxel_grid(events, num_bins=5, width=max(events[:, 1]) + 1, height=max(events[:, 2]) + 1, is_hard=is_hard)

        """Numpy to Tensor"""
        voxel_grid = torch.tensor(voxel_grid)

        """Numpy to Tensor"""
        voxel_grid = normalize_voxel(voxel_grid)

        """Resize Tensor"""
        voxel_grid = F.interpolate(voxel_grid.unsqueeze(0).float(), size=(384, 192), mode='nearest').squeeze(0)

        label = self.label_list[index]
        cam = self.cam_list[index]
        acculabel = self.accumulate_labels[index]

        return voxel_grid, label, cam, acculabel, index, is_hard

    def __len__(self):
        return len(self.name_list)

    def get_index_dict(self):
        index_dict = defaultdict(list)
        index = 0
        for i in self.label_list:
            index_dict[i].append(index)
            index += 1
        return index_dict


class voxelDatasetcam_new2(Dataset):

    def __init__(self, pure_dataset=None, event_dir="data/", mode='train'):
        self.name_list = pure_dataset.name_list
        self.cam_list = pure_dataset.cam_list
        self.label_list = pure_dataset.label_list
        self.event_dir = event_dir
        self.mode = mode
        self.relabel_under_eachcam()

    def relabel_under_eachcam(self):
        all_label = np.array(self.label_list)
        all_cams = np.array(self.cam_list)
        self.accumulate_labels = np.zeros(all_label.shape, all_label.dtype)
        prev_id_count = 0
        self.id_count_each_cam = []
        for this_cam in np.unique(all_cams):
            percam_labels = all_label[all_cams == this_cam]
            unique_id = np.unique(percam_labels)
            self.id_count_each_cam.append(len(unique_id))
            id_dict = {ID: i for i, ID in enumerate(unique_id.tolist())}
            for i in range(len(percam_labels)):
                percam_labels[i] = id_dict[percam_labels[i]]  # relabel
            self.accumulate_labels[all_cams == this_cam] = percam_labels + prev_id_count
            prev_id_count += len(unique_id)
        print('  sum(id_count_each_cam)= {}'.format(sum(self.id_count_each_cam)))

    def __getitem__(self, index):  # 一个event chunk一个voxel grid?
        events = np.loadtxt(self.event_dir + self.mode + '/' + self.name_list[index], dtype=np.float64,
                            delimiter=' ',
                            usecols=(0, 1, 2, 3))
        """ 640x480 => bbox(x_, y_), as we have already clean/removed the events outside bbox"""
        events[:, 1] = events[:, 1] - min(events[:, 1])  # x - x_min
        events[:, 2] = events[:, 2] - min(events[:, 2])  # y - y_min
        events[:, 0] = (events[:, 0] - min(events[:, 0])) * 1000  # t - t_min

        """event => voxel-grid"""
        voxel_grid = events_to_voxel_grid(events, num_bins=5, width=max(events[:, 1]) + 1, height=max(events[:, 2]) + 1)

        """Numpy to Tensor"""
        voxel_grid = torch.tensor(voxel_grid)

        """Numpy to Tensor"""
        voxel_grid = normalize_voxel(voxel_grid)

        """Resize Tensor"""
        voxel_grid = F.interpolate(voxel_grid.unsqueeze(0).float(), size=(384, 192), mode='nearest').squeeze(0)

        label = self.label_list[index]
        cam = self.cam_list[index]
        acculabel = self.accumulate_labels[index]

        return voxel_grid, label, cam, acculabel, index

    def __len__(self):
        return len(self.name_list)

    def get_index_dict(self):
        index_dict = defaultdict(list)
        index = 0
        for i in self.label_list:
            index_dict[i].append(index)
            index += 1
        return index_dict

if __name__ == '__main__':
    pass
