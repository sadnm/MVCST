import os
import torch
import random
import numpy as np
import scipy.sparse as sp
from matplotlib.font_manager import FontProperties
from torch.backends import cudnn
import networkx as nx
import pandas as pd


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)

def permutation(feature):
    np.random.seed(2024)
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]

    return feature_permutated


def concat_adj(matrix1,matrix2):
    n1, m1 = matrix1.shape
    n2, m2 = matrix2.shape
    adj_1_right = sp.hstack([matrix1, sp.csr_matrix((n1, m2))])  # adj_1右边填充零矩阵
    adj_2_left = sp.hstack([sp.csr_matrix((n2, m1)), matrix2])  # adj_2左边填充零矩阵

    adj_combined = sp.vstack([adj_1_right, adj_2_left])  # 将两个矩阵垂直拼接
    return adj_combined

def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    print("n_spot: ", n_spot)
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    return label_CSL

# https://github.com/ClayFlannigan/icp
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    # assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def ICP_align(adata_target, adata_ref, prob_matrix, plot_align=False):
    """
    使用概率配对矩阵对两个切片进行刚体 ICP 配准。

    Parameters:
        adata_target : AnnData
            待对齐的切片 AnnData（源）
        adata_ref : AnnData
            参考切片 AnnData（目标）
        prob_matrix : np.ndarray
            概率对应矩阵，形状 (N_target_spots, N_ref_spots)，行为源点，列为参考点
        plot_align : bool
            是否画对齐前后图

    Returns:
        aligned_coords : np.ndarray
            对齐后的源切片空间坐标，形状 (N_target_spots, 2)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from copy import deepcopy

    # 提取原始空间坐标
    coor_src = adata_target.obsm["spatial"]
    coor_dst = adata_ref.obsm["spatial"]

    # 根据概率矩阵计算每个源点的 soft 对应目标点位置
    weighted_dst = prob_matrix @ coor_dst  # 形状: [N_src, 2]

    # 进行刚体变换拟合：src -> weighted_dst
    T, _, _ = best_fit_transform(coor_src, weighted_dst)

    # 应用变换到整个切片（齐次坐标形式）
    coor_src_homo = np.concatenate([coor_src, np.ones((coor_src.shape[0], 1))], axis=1).T  # shape: (3, N)
    aligned_coords = (T @ coor_src_homo).T[:, :2]  # 转换回来 shape: (N, 2)

    # 可视化对齐前后
    if plot_align:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(coor_dst[:, 0], coor_dst[:, 1], c='gray', s=2, label='Reference')
        plt.scatter(coor_src[:, 0], coor_src[:, 1], c='blue', s=2, label='Target (Before)')
        plt.title("Before Alignment")
        plt.axis('equal');
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(coor_dst[:, 0], coor_dst[:, 1], c='gray', s=2, label='Reference')
        plt.scatter(aligned_coords[:, 0], aligned_coords[:, 1], c='green', s=2, label='Target (Aligned)')
        plt.title("After Alignment")
        plt.axis('equal');
        plt.legend()
        plt.tight_layout()
        plt.show()

    return aligned_coords

import matplotlib.pyplot as plt
def visualize_3d_slices_matplotlib(slices, z_scale=2.0, cluster_key='clust', landmark=None):
    """
    使用 matplotlib 绘制多个切片的 3D 空间可视化图（优化版：颜色更深，点更清晰）。

    Parameters
    ----------
    slices : list of AnnData
        已配准的多个切片对象。
    z_scale : float
        每一层切片之间的 z 坐标间隔。
    cluster_key : str
        obs 中用于上色的列名。
    landmark : str or int or None
        如果指定，将该类设置为高亮显示。
    """
    from matplotlib import rcParams

    # 设置字体为 Times New Roman
    rcParams['font.family'] = 'SimSun'
    rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(10, 8),dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    default_colors = ['#b2cde3','#3e70a0','#b5dd8b','#55a339','#ed9e9a','#c93331','#f3c072','#c3b3cc','#5f428b']

    for i, adata in enumerate(slices):
        if 'spatial' not in adata.obsm:
            print(f"❗ 跳过第 {i} 个切片：未找到 obsm['spatial']")
            continue

        coor = adata.obsm['spatial']
        z = np.ones((coor.shape[0], 1)) * i * z_scale
        coor_3d = np.hstack([coor, z])

        labels = adata.obs[cluster_key].astype(str).values
        unique_labels = np.unique(labels)

        # 颜色字典
        color_dict = {}
        if f"{cluster_key}_colors" in adata.uns:
            lut = adata.uns[f"{cluster_key}_colors"]
            for idx, l in enumerate(unique_labels):
                color_dict[l] = lut[idx % len(lut)]
        else:
            for idx, l in enumerate(unique_labels):
                color_dict[l] = default_colors[idx % len(default_colors)]

        # 分组画图
        for label in unique_labels:
            mask = labels == label
            xyz = coor_3d[mask]

            ax.scatter(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                c=color_dict[label],
                s=16 if label == str(landmark) else 14,
                alpha=1.0 if label == str(landmark) else 0.8,
                label=label if i == 0 else None
            )

    ax.view_init(elev=20, azim=-40)
    ax.set_title("三维坐标对齐", fontsize=20, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    ax.legend(
        title='',
        bbox_to_anchor=(1.08, 0.5),
        loc='right',
        frameon=False,
        markerscale=2,
        prop = FontProperties(family='Times New Roman')
    )

    plt.tight_layout()
    plt.savefig("人类心脏三维坐标对齐.pdf")



def visualize_2d_alignment_overlay_by_cluster(
    adata1,
    adata2,
    cluster_key='clust',
    alpha=1.0,
    landmark_indices=None,
    landmark_color='red',
    font='Times New Roman',
    figsize=(6, 6)
):
    """
    可视化两个 AnnData 对象在二维空间对齐后的重叠效果，按 cluster_key 分类上色。

    Parameters
    ----------
    adata1, adata2 : AnnData
        包含对齐后的坐标 (obsm['spatial']) 和分类信息 (obs[cluster_key])。
    cluster_key : str
        obs 中表示类别的列名。
    alpha : float
        点透明度。
    landmark_indices : list of int
        要高亮的 landmark 索引。
    landmark_color : str
        高亮颜色。
    font : str
        字体名称。
    figsize : tuple
        图尺寸。
    """

    default_colors = ['#b2cde3', '#3e70a0', '#b5dd8b', '#55a339', '#ed9e9a',
                      '#c93331', '#f3c072', '#c3b3cc', '#5f428b', '#9370DB']

    fig, ax = plt.subplots(figsize=figsize)
    font_prop = FontProperties(family=font)

    coor1 = adata1.obsm['spatial']
    coor2 = adata2.obsm['spatial']
    labels = adata1.obs[cluster_key].astype(str).values  # 假设两个 adata 已对齐，label 是一致的

    unique_labels = sorted(np.unique(labels))
    color_dict = {label: default_colors[i % len(default_colors)] for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        mask1 = adata1.obs[cluster_key].astype(str).values == label
        mask2 = adata2.obs[cluster_key].astype(str).values == label
        ax.scatter(coor1[mask1, 0], coor1[mask1, 1], c=color_dict[label], label=f"{label} (1)", alpha=alpha, s=25)
        ax.scatter(coor2[mask2, 0], coor2[mask2, 1], c=color_dict[label], label=f"{label} (2)", alpha=alpha * 0.6, s=25, marker='x')

    if landmark_indices is not None:
        ax.scatter(coor1[landmark_indices, 0], coor1[landmark_indices, 1],
                   c=landmark_color, s=40, label='Landmark', edgecolors='black')
        ax.scatter(coor2[landmark_indices, 0], coor2[landmark_indices, 1],
                   c=landmark_color, s=40, edgecolors='black')

    ax.set_xlabel("spatial1", fontproperties=font_prop)
    ax.set_ylabel("spatial2", fontproperties=font_prop)
    ax.set_title("对齐后重叠图（按类别）", fontproperties=font_prop, fontsize=14, pad=20)
    ax.legend(prop=font_prop, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.axis('equal')
    plt.tight_layout()
    plt.show()

