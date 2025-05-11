import numpy as np
import torch
import copy
from osgeo import gdal
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler
from torch.utils.data import Dataset
from collections import Counter

from utils.preprocess import Processor
from utils.superpixel_utils import HSI_to_superpixels, create_association_mat, \
    create_spixel_graph, show_superpixel, extract_superpixel_features



def get_sp_labels(seg, gt):
    labels = np.zeros((np.unique(seg).shape[0]), dtype=np.int32)
    for i in range(len(labels)):
        mask = seg == i
        label = gt[mask]
        c = Counter(label).most_common()  # 出现频率最多的标签
        if c[0][0] == 0 and len(c) > 1:
            labels[i] = c[1][0]  # 0是背景,计算精度会忽略
        else:
            labels[i] = c[0][0]
    return labels


class HSIDataset(Dataset):
    def __init__(self, image_path, gt_path, clip=None, seg_path: str = None, neighbors=50, patch_size=7, n_superpixels=500, pca=True,
                 pca_dim=8):
        self.n_superpixels = n_superpixels
        self.name = image_path.split('/')[-1].split('.')[0]
        self.neighbors = neighbors
        p = Processor()
        img, gt = p.prepare_data(image_path, gt_path)
        img_copy = copy.deepcopy(img)
        n_row_org, n_col_org, n_band_org = img.shape
        if clip is not None:
            img = img[clip[0][0]:clip[0][1], clip[1][0]:clip[1][1], :]
            gt = gt[clip[0][0]:clip[0][1], clip[1][0]:clip[1][1]]
        self.img = img
        self.gt = gt
        self.gt = p.standardize_label(self.gt)
        n_row, n_col, n_band = img.shape

        scaler = StandardScaler()
        img = scaler.fit_transform(img.reshape(-1, n_band))
        # img = scale(img.reshape(n_row * n_col, -1))

        if pca:
            pca = PCA(n_components=pca_dim)
            img = pca.fit_transform(img).reshape(n_row, n_col, -1)

        if seg_path is not None:
            if seg_path.endswith('.mat'):
                self.sp_labels = loadmat(seg_path)['labels']
            elif seg_path.endswith('.tif'):
                tif_file = gdal.Open(seg_path)
                self.sp_labels = tif_file.GetRasterBand(1).ReadAsArray(0, 0, n_col_org,
                                                                       n_row_org)  # .reshape(1, n_row, n_col)

            if clip is not None:
                self.sp_labels = self.sp_labels[clip[0][0]:clip[0][1], clip[1][0]:clip[1][1]]
            self.sp_labels = p.standardize_label(self.sp_labels)

        else:
            self.sp_labels = HSI_to_superpixels(img, n_superpixels=self.n_superpixels, is_show_superpixel=False)

        show_superpixel(self.sp_labels, img_copy[:, :, :3],
                        'result/' + self.name + '.pdf')  # 需要特定通道在第三维上修改，如img[:, :, (10, 20, 30)]

        self.association_mat = create_association_mat(self.sp_labels)
        self.sp_graph, self.sadj, self.sp_centroid = create_spixel_graph(img, self.sp_labels, self.neighbors)

        for i in np.unique(self.gt):
            print('class: {:02d}, samples: {}'.format(i, np.nonzero(self.gt == i)[0].shape[0]))
        self.n_classes = np.unique(self.gt).shape[0] - 1  # 去掉背景0

        self.patch_sp_features = extract_superpixel_features(img, self.sp_labels, mode='center_patch',
                                                             patch_size=patch_size)  # 取patch的超像素特征
        self.sp_features = extract_superpixel_features(img, self.sp_labels, mode='mean_std_geo')  # 取像素平均的超像素特征
        self.labels = get_sp_labels(self.sp_labels, self.gt)

        self.association_mat = torch.from_numpy(self.association_mat).type(torch.FloatTensor)
        self.association_mat_copy = self.association_mat[:]
        self.sp_graph = torch.from_numpy(self.sp_graph).type(torch.FloatTensor)
        # self.sadj = torch.from_numpy(self.sadj).type(torch.FloatTensor)
        self.patch_sp_features = torch.from_numpy(self.patch_sp_features).type(torch.FloatTensor)
        self.sp_features = torch.from_numpy(self.sp_features).type(torch.FloatTensor)
        self.img = torch.from_numpy(img.transpose(2, 0, 1)).type(torch.FloatTensor)

    def remove_background(self):  # 去掉超像素标签为0的超像素，以防干扰
        non_zero_idx = np.nonzero(self.labels)[0]
        self.association_mat = self.association_mat[:, non_zero_idx]
        self.sp_graph = self.sp_graph[non_zero_idx, :][:, non_zero_idx]
        # self.sadj = self.sadj[non_zero_idx, :][:, non_zero_idx]
        self.patch_sp_features = self.patch_sp_features[non_zero_idx, :]
        self.sp_features = self.sp_features[non_zero_idx, :]

    def recover_background(self, sp_pred):
        if isinstance(sp_pred, torch.Tensor):
            sp_pred = sp_pred.detach().cpu().numpy()
        non_zero_idx = np.nonzero(self.labels)[0]
        sp_pred_full = np.zeros_like(self.labels)
        sp_pred_full[non_zero_idx] = sp_pred
        self.association_mat = self.association_mat_copy
        return sp_pred_full

    def __getitem__(self, idx):
        x1 = self.patch_sp_features[idx]
        x2 = self.sp_features[idx]
        return x1, x2

    def __len__(self):
        return self.sp_features.shape[0]

    def __str__(self) -> str:
        string = '-----------------------------------------------------\n'
        string += '| Dataset: {}\n'.format(self.name)
        string += '-----------------------------------------------------\n'
        string += '| Number of real sps\t: {}\n'.format(np.where(self.labels != 0)[0].shape[0])
        string += '| Number of superpixels\t: {}\n'.format(self.sp_features.shape[0])
        string += '| Number of classes\t: {}\n'.format(self.n_classes)
        string += '| Patch feature (sp)\t: {}\n'.format(self.patch_sp_features.shape)
        string += '| Px_mean feature (sp)\t: {}\n'.format(self.sp_features.shape)
        string += '-----------------------------------------------------\n'
        return string

    def __repr__(self) -> str:
        return self.__str__()


if __name__ == '__main__':
    dataset = HSIDataset("../HSI_data/Indian_pines.mat", "../HSI_data/Indian_pines_gt.mat", None, None, patch_size=7,
                         n_superpixels=500)
    print(dataset)