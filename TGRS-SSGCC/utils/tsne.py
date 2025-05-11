import matplotlib.pyplot as plt
import numpy as np

import os
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import shuffle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.ticker as ticker
import seaborn as sns


def plot_latent_embedding(train_z, all_labels, method_name, dataset, ACC):
    seed = 0
    all_labels = all_labels[all_labels != 0]
    ts = TSNE(n_components=2, init='random', random_state=seed).fit_transform(train_z)

    train_fts_labels = all_labels
    domain_num = max(train_fts_labels)
    reduced_z = ts

    # plot loss curve
    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'bold',
            'size': 7,
            }
    mycolor = np.array([[224, 32, 32],
                        [255, 192, 0],
                        [32, 160, 64],
                        [48, 96, 192],
                        [192, 48, 192]]) / 255.0
    mymarker = ['1', '2', 's', '*', 'H', 'D', 'o', '>']

    palette = np.array(sns.color_palette('hls', domain_num))

    my_line_width = 3
    my_marker_size = 10

    label_z_dict = {}
    # print(train_fts_labels)
    for label in list(set(train_fts_labels)):
        z_for_label = reduced_z[np.where(train_fts_labels == label)]
        label_z_dict[label] = z_for_label
    # print(label_z_dict)

    color_list = [(95 / 255, 80 / 255, 164 / 255),
                  (72 / 255, 142 / 255, 190 / 255),
                  (126 / 255, 199 / 255, 164 / 255),
                  (252 / 255, 237 / 255, 157 / 255),  # 1
                  (244 / 255, 253 / 255, 168 / 255),
                  (193 / 255, 229 / 255, 157 / 255),  # 1-2
                  (223 / 255, 174 / 255, 102 / 255),
                  (238 / 255, 121 / 255, 66 / 255),
                  (207 / 255, 65 / 255, 76 / 255),
                  (151 / 255, 0 / 255, 65 / 255),
                  (116 / 255.0, 116 / 255.0, 241 / 255.0),
                  (238 / 255.0, 91 / 255.0, 154 / 255.0),
                  (208 / 255.0, 189 / 255.0, 255 / 255.0),
                  (252 / 255.0, 163 / 255.0, 203 / 255.0),
                  (247 / 255.0, 171 / 255.0, 121 / 255.0),
                  (203 / 255.0, 153 / 255.0, 126 / 255.0)
                  ]



    plt.figure(1)
    xmax, xmin, ymax, ymin = 0, 0, 0, 0
    for label in label_z_dict:
        points = label_z_dict[label]
        xmax = max(max(label_z_dict[label][:, 0]), xmax)
        xmin = min(min(label_z_dict[label][:, 0]), xmin)
        ymax = max(max(label_z_dict[label][:, 1]), ymax)
        ymin = min(min(label_z_dict[label][:, 1]), ymin)
        for point in points:
            x, y = point
            plt.text(x, y, str(int(label)-1), color=color_list[int(label)-1], fontdict=font)
        # plt.text(label_z_dict[label][:, 0], label_z_dict[label][:, 1], str(label_z_dict[label]), color=color_list[label_z_dict[label]], fontdict=font)
    # plt.show()
    plt.xlim(xmin - 5, xmax + 5)
    plt.ylim(ymin - 5, ymax + 5)
    ax = plt.subplot(111)  # 创建子图
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./{}_{}_{}_NMI={}new.png'.format(dataset, method_name, seed, ACC), dpi=600)
    plt.close('all')


#     train_fts_labels = all_labels
#     domain_num = max(train_fts_labels)+1
#     reduced_z = TSNE(n_components=2, init='random',random_state=seed).fit_transform(train_z)


#     # plot loss curve
#     font = {'family': 'Times New Roman',
#             'color': 'black',
#             'weight': 'bold',
#             'size': 15,
#             }
#     mycolor = np.array([[224, 32, 32],
#                             [255, 192, 0],
#                             [32, 160, 64],
#                             [48, 96, 192],
#                             [192, 48, 192]]) / 255.0
#     mymarker = ['1', '2', 's', '*', 'H', 'D', 'o', '>']

#     palette = np.array(sns.color_palette('hls', domain_num))

#     my_line_width = 3
#     my_marker_size = 10

#     label_z_dict = {}
#     for label in list(set(train_fts_labels)):
#         z_for_label = reduced_z[np.where(train_fts_labels==label)]
#         label_z_dict[label] = z_for_label

#     plt.figure(1)
#     for label in label_z_dict:
#         plt.scatter(label_z_dict[label][:, 0], label_z_dict[label][:, 1], s=20, color=palette[label])
#     # plt.show()
#     plt.savefig(os.path.join(os.getcwd(), 'figure', '{}_{}_tsne_random_train_{}.png'.format(dataset, method_name, seed)))
#     plt.close('all')


def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure(dpi=600)  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        # 颜色盘
        color_list = [
            (95 / 255, 80 / 255, 164 / 255),
            (72 / 255, 142 / 255, 190 / 255),
            (126 / 255, 199 / 255, 164 / 255),
            (193 / 255, 229 / 255, 157 / 255),
            (244 / 255, 253 / 255, 168 / 255),
            (252 / 255, 237 / 255, 157 / 255),
            (223 / 255, 174 / 255, 102 / 255),
            (238 / 255, 121 / 255, 66 / 255),
            (207 / 255, 65 / 255, 76 / 255),
            (151 / 255, 0 / 255, 65 / 255)
        ]
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=color_list[label[i]],
                 fontdict={'weight': 'bold', 'size': 7})
    plt.xticks([])  # 指定坐标的刻度
    plt.yticks([])
    plt.grid()

    plt.savefig('./{}.jpeg'.format(title))
    # plt.show()
    # plt.xticks(np.linspace(-0.1, 1.1, 10))
    # plt.yticks(np.linspace(0, 1.1, 5))
    # plt.title(title, fontsize=14)
    # plt.xlabel('Factors')
    # plt.ylabel('Eigenvalue')
    # 返回值
    # return fig

# embeds = model.embed(graph.to(device), x.to(device)).detach().cpu().numpy() # 所学的表示
# embeds = x.numpy()  # 原始属性矩阵
# label = graph.ndata["label"].numpy()  # 原始标签
# from sklearn.manifold import TSNE
# ts = TSNE(n_components=2, init='pca', random_state=0)
# ts = TSNE(n_components=2, init='random', random_state=2)
# result = ts.fit_transform(embeds[:2500, :])   # 取2500个样本
# fig = plot_embedding(result, label[:2500], 't-SNE Embedding of digits')
# plt.savefig(f'tsne/{dataset_name}_raw_tsne_{seed}.png')