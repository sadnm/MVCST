import scanpy as sc
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph


# 数据预处理
def preprocess(adata):
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)##2000
    # sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata[:, adata.var['highly_variable']]##对基因额外的处理方式

# 空间位置图
def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=50, model='Radius', verbose=True):
    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    # X = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)

    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    # G = G.toarray()
    G = G + G.T
    G.data = np.minimum(G.data, 1)
    # G = preprocess_graph(G)
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.obsm['adj'] = G
    print(G)
    # adata.uns['adj'] = G

def Cal_Feature_Net(adata, k=20, mode= "distance", metric="minkowski", include_self=False):##以距离对基因表达数据进行图建模
    print("Begin calculate feature graph")
    feature_graph = kneighbors_graph(adata.obsm['feat_pca'], k, mode=mode, metric=metric,
                                            include_self=include_self)
    feature_graph = feature_graph+feature_graph.T
    feature_graph.data = np.minimum(feature_graph.data, 1)
    # feature_graph = preprocess_graph(feature_graph)
    adata.obsm['feature_adj'] = feature_graph
    print(feature_graph)


# 预处理邻接矩阵
def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])  # 添加自环
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized


# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)

def concat_adj(adj_1,adj_2):
    n1, m1 = adj_1.shape
    n2, m2 = adj_2.shape
    adj_1_right = sp.hstack([adj_1, sp.csr_matrix((n1, m2))])  # adj_1右边填充零矩阵
    adj_2_left = sp.hstack([sp.csr_matrix((n2, m1)), adj_2])  # adj_2左边填充零矩阵

    adj_combined = sp.vstack([adj_1_right, adj_2_left])  # 将两个矩阵垂直拼接
    return adj_combined


def pca(adata, use_reps=None, n_comps=50):
    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)

    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    adata.obsm['feat_pca'] = feat_pca
