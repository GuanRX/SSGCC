import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, minmax_scale, normalize, StandardScaler


from matplotlib import cm
import matplotlib as mpl
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import cv2
import typing
# from skimage.future import graph

def HSI_to_superpixels(img, n_superpixels, is_show_superpixel=False):
    superpixel_label = slic(img, n_segments=n_superpixels, compactness=10,  convert2lab=False,
                            enforce_connectivity=True, min_size_factor=0.3, max_size_factor=2, slic_zero=False, start_label=0)
    if is_show_superpixel:
        show_superpixel(superpixel_label, img)
    return superpixel_label


def show_superpixel(label, x=None, savepath='superpixels.pdf'):
    color = (162 / 255, 169 / 255, 175 / 255)
    n_row, n_col, n_band = x.shape
    if x is not None:
        x = minmax_scale(x.reshape(n_row * n_col, n_band)).reshape(n_row, n_col, n_band)
        mask = mark_boundaries(x, label, color=(1, 1, 0), mode='subpixel')
    else:
        mask_boundary = find_boundaries(label, mode='subpixel')
        mask = np.ones((n_row, n_col, 3))
        mask[mask_boundary] = color
    fig = plt.figure()
    plt.imshow(mask)
    plt.axis('off')
    plt.tight_layout()
    fig.savefig(savepath, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


def create_association_mat(superpixel_labels):
    labels = np.unique(superpixel_labels)
    # print(labels)
    n_labels = labels.shape[0]
    n_pixels = superpixel_labels.shape[0] * superpixel_labels.shape[1]
    association_mat = np.zeros((n_pixels, n_labels))
    superpixel_labels_ = superpixel_labels.reshape(-1)
    for i, label in enumerate(labels):
        association_mat[np.where(label == superpixel_labels_), i] = 1
    return association_mat


def create_spixel_graph(source_img, superpixel_labels, neighbors):
    s = source_img.reshape((-1, source_img.shape[-1]))
    a = create_association_mat(superpixel_labels)
    # t = superpixel_labels.reshape(-1)
    mean_fea = np.matmul(a.T, s)
    regions = regionprops(superpixel_labels + 1)
    n_labels = np.unique(superpixel_labels).shape[0]
    center_indx = np.zeros((n_labels, 2))
    for i, props in enumerate(regions):
        center_indx[i, :] = props.centroid  # centroid coordinates
    ss_fea = np.concatenate((mean_fea, center_indx), axis=1)
    ss_fea = minmax_scale(ss_fea)
    try:
        adj = kneighbors_graph(ss_fea, n_neighbors=neighbors, mode='distance', include_self=False).toarray()
    except:
        adj = kneighbors_graph(ss_fea, n_neighbors=np.unique(superpixel_labels).shape[0] // 2, mode='distance', include_self=False).toarray()

    # # # show initial graph
    # import matplotlib.pyplot as plt
    # adj_ = np.copy(adj)
    # adj_[np.where(adj != 0)] = 1
    # plt.imshow(adj_, cmap='hot')
    # plt.show()

    # # auto calculate gamma in Gaussian kernel
    X_var = ss_fea.var()
    gamma = 1.0 / (ss_fea.shape[1] * X_var) if X_var != 0 else 1.0
    adj[np.where(adj != 0)] = np.exp(-np.power(adj[np.where(adj != 0)], 2) * gamma)

    # adj = euclidean_dist(ss_fea, ss_fea).numpy()
    # adj = np.exp(-np.power(adj, 2) * gamma)
    np.fill_diagonal(adj, 0)

    # show_graph(adj, center_indx)
    # 从superpixel_labels生成超像素的拓扑图
    # g = graph.RAG(superpixel_labels)
    # sadj = np.array(nx.linalg.adjacency_matrix(g).todense())
    # sadj = sadj + np.eye(sadj.shape[0])
    return adj, None, center_indx
    # return adj, center_indx


def extract_superpixel_features(img, labels, mode:typing.Literal['mean_std_geo', 'mean_std', 'center_patch']='mean_std_geo', patch_size=None):
    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    """
    对输入图像进行超像素特征提取,并返回每个超像素的特征向量。

    Args:
        img (numpy.ndarray): 输入图像,可以是任意尺寸和深度的彩色图像。
        labels (numpy.ndarray): 输入图像的超像素分割结果,每个像素的值为所属超像素的标号。
        mode (str): 特征提取方法
        patch_size (int): 如果mode为center_patch则用到,中心取patch

    Returns:
        numpy.ndarray: 每个超像素的特征向量,包含各波段的平均值和方差、超像素的长宽比、超像素的面积、超像素的周长。
    """
    num_labels = np.unique(labels).shape[0]                         # 获取超像素数量
    if mode == 'mean_std_geo':
        features = np.zeros((num_labels, img.shape[2] * 2 + 3), dtype='float32')  # 初始化特征向量矩阵

        # 计算每个超像素的特征
        for i in range(num_labels):
            # 获取当前超像素的掩码
            mask = np.zeros_like(labels, dtype='uint8')
            mask[labels==i] = 1

            mean = np.mean(img[labels==i], axis=0)
            std = np.std(img[labels==i], axis=0)
            features[i, 0:img.shape[2]]        = mean
            features[i, img.shape[2]:img.shape[2] * 2] = std
            
            # 计算超像素的面积、周长和长宽比
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = cv2.arcLength(contours[0], True)
            area = cv2.contourArea(contours[0])
            aspect_ratio = float(np.max(contours[0][:, 0, 0]) - np.min(contours[0][:, 0, 0])) / \
                        float(np.max(contours[0][:, 0, 1]) - np.min(contours[0][:, 0, 1]) + 1)
            if aspect_ratio > 1:
                aspect_ratio = 1 / aspect_ratio
            features[i, -3] = area
            features[i, -2] = perimeter
            features[i, -1] = aspect_ratio

        regions = regionprops(labels + 1)
        n_labels = np.unique(labels).shape[0]
        center_indx = np.zeros((n_labels, 2))
        for i, props in enumerate(regions):
            center_indx[i, :] = props.centroid  # centroid coordinates
        features = np.concatenate((center_indx, features), axis=1)

        # 分区域归一化
        features[:, :2] = normalization(features[:, :2])
        features[:, 2:-3] = normalization(features[:, 2:-3])
        features[:, -3] = normalization(features[:, -3])
        features[:, -2] = normalization(features[:, -2])

    elif mode == 'mean_std':
        features = np.zeros((num_labels, img.shape[2] * 2), dtype='float32')  # 初始化特征向量矩阵

        # 计算每个超像素的特征
        for i in range(num_labels):
            # 获取当前超像素的掩码
            # mask = np.zeros_like(labels, dtype='uint8')
            # mask[labels==i] = 1

            mean = np.mean(img[labels==i], axis=0)
            std = np.std(img[labels==i], axis=0)
            features[i, 0:img.shape[2]]                = mean
            features[i, img.shape[2]:img.shape[2] * 2] = std
            
            # # 计算超像素的面积、周长和长宽比
            # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # perimeter = cv2.arcLength(contours[0], True)
            # area = cv2.contourArea(contours[0])
            # aspect_ratio = float(np.max(contours[0][:, 0, 0]) - np.min(contours[0][:, 0, 0])) / \
            #             float(np.max(contours[0][:, 0, 1]) - np.min(contours[0][:, 0, 1]) + 1)
            # if aspect_ratio > 1:
            #     aspect_ratio = 1 / aspect_ratio
            # features[i, -3] = area
            # features[i, -2] = perimeter
            # features[i, -1] = aspect_ratio


        regions = regionprops(labels + 1)
        n_labels = np.unique(labels).shape[0]
        center_indx = np.zeros((n_labels, 2))
        for i, props in enumerate(regions):
            center_indx[i, :] = props.centroid  # centroid coordinates
        features = np.concatenate((center_indx, features), axis=1)

        # # 分区域归一化
        features[:, :2] = normalization(features[:, :2])
        # features[:, 2:-3] = normalization(features[:, 2:-3])
        # features[:, -3] = normalization(features[:, -3])
        # features[:, -2] = normalization(features[:, -2])

    elif mode == 'center_patch':    # 从超像素中心点取patch
        assert patch_size % 2 == 1, "Patch size must be odd"
        features = np.zeros((num_labels, patch_size, patch_size, img.shape[2]), dtype='float32')  # 初始化特征向量矩阵
        regions = regionprops(labels + 1)
        # center_indx = np.zeros((np.unique(labels).shape[0], 2))
        pad_size = patch_size // 2
        # center_indx += pad_size
        pad_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
        for i, props in enumerate(regions):
            center = props.centroid  # centroid coordinates
            x, y = center
            x = int(x + pad_size)
            y = int(y + pad_size)
            feat = pad_img[x - pad_size:x + pad_size + 1, y - pad_size:y + pad_size + 1, :]
            features[i, :] = feat
        features = np.transpose(features, (0, 3, 1, 2))  # 转置为[n, c, h, w]
        # features = normalization(features)

    # 返回特征向量矩阵
    return features


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    import torch
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-8).sqrt()  # for numerical stability
    return dist

# def cosine_sim_with_temperature(x, temperature=0.5):
#     x = normalize(x)
#     sim = np.matmul(x, x.T) / temperature  # Dot similarity

def show_graph(adj, node_pos):
    plt.style.use('seaborn-white')
    D = np.diag(np.reshape(1./np.sum(adj, axis=1), -1))
    adj = np.dot(D, adj)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    G = nx.from_numpy_array(adj)
    for i in range(node_pos.shape[0]):
        G.nodes[i]['X'] = node_pos[i]
    # edge_weights = [(u, v) for u, v in G.edges()]
    pos = nx.get_node_attributes(G, 'X')
    # nx.draw(G, pos=pos, node_size=40, node_color='b', edge_color='black')  #  #fabebe # white
    nx.draw(G, pos=pos, node_size=40, node_color='#CD3700')  # #fabebe # white
    norm_v = mpl.colors.Normalize(vmin=0, vmax=adj.max())
    cmap = cm.get_cmap(name='PuBu')
    m = cm.ScalarMappable(norm=norm_v, cmap=cmap)
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               width=d['weight'] * 5, alpha=0.5, edge_color=m.to_rgba(d['weight']))
    # draw graph
    # nx.draw(G, pos=pos, node_size=40)
    # nx.draw(G, pos=pos, node_size=node_size, node_color=color)
    plt.show()


def choose_pixels(sp_labels, num_pixels):
    """
    从超像素分割结果中随机提取每个超像素的 n 个像素的坐标
    :param slic_result: 超像素分割结果,大小为M*N,每个元素为超像素的标签
    :param n: 每个超像素提取的像素数
    :return: 大小为K*n*2的矩阵,其中K为超像素数量,n为每个超像素提取的像素数
    """
    unique_labels = np.unique(sp_labels)
    sp_labels_1d = sp_labels.reshape(-1)

    result = np.zeros((unique_labels.size, num_pixels), dtype=np.int32)
    for i, label in enumerate(unique_labels):
        indices = np.where(sp_labels_1d == label)
        # points = np.column_stack((indices[0], indices[1]))
        random_indices = np.random.choice(indices[0].size, num_pixels, replace=True)
        result[i] = indices[0][random_indices]

    return result

