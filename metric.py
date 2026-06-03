# -*- coding: UTF-8 -*-


import scipy
import numpy as np
import pandas as pd
import networkx as nx
import ot
from scib.metrics.lisi import lisi_graph_py
from tqdm import trange
import scanpy as sc
import sklearn.neighbors
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 聚类指标
def mclust_R(adata, num_cluster, modelNames="EEE", used_obsm='emb_pca', random_seed=2024):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    print(res)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01,
               refinement=False):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'.
    start : float
        The start value for searching. The default is 0.1.
    end : float
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """
    pca = PCA(n_components=50, random_state=2024)
    embedding = pca.fit_transform(adata.obsm[key].copy())
    adata.obsm['emb_pca'] = embedding

    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'kmeans':
        pca = PCA(n_components=20, random_state=2024)
        embedding = pca.fit_transform(adata.obsm[key].copy())
        kmeans = KMeans(n_clusters=n_clusters).fit(embedding)
        kmeans_result = [i + 1 for i in kmeans.labels_]
        adata.obs['domain'] = list(map(lambda x: str(x), kmeans_result))
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain']

    if refinement:
        # pass
        new_type = refine_label(adata, radius, key='domain')
        adata.obs['domain'] = new_type


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res


def nolabel_clustering_matrix(X, labels):
    sc_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)

    print("sc_score", sc_score)
    print("ch_score", ch_score)
    print("db_score", db_score)

def _get_spatial_entropy(C, C_sum):
    H = 0
    for i in range(len(C)):
        for j in range(len(C)):
            z = C[i, j]
            if z != 0:
                H += -(z / C_sum) * np.log(z / C_sum)
    return H

def spatial_entropy(k_neighbors, labels, degree=4):
    """
    Calculates spatial entropy of graph
    """
    # construct contiguity matrix C which counts pairs of cluster edges
    # nx.set_node_attributes(g, labels, "labels")
    # cluster_nums = len(np.unique(list(labels.values())))
    # C = np.zeros((cluster_nums, cluster_nums))
    # for e in g.edges():
    #     C[labels[e[0]]][labels[e[1]]] += 1


    # S = np.repeat(adata.obs[annotation_key].values[:, None], degree, axis=1)
    S = np.broadcast_to(labels[:, None], (len(labels), degree))
    N = labels[k_neighbors]
    cluster_names = np.unique(labels)
    cluster_nums = len(cluster_names)
    C = np.zeros((cluster_nums, cluster_nums))
    for i in range(cluster_nums):
        for j in range(cluster_nums):
            # C[i, j] = np.sum(np.logical_and(N == i, S == j))
            C[i, j] = np.sum(np.logical_and(S == cluster_names[i], N == cluster_names[j]))
    # cluster_names = np.unique(list(labels.values()))
    # C = pd.DataFrame(0,index=cluster_names, columns=cluster_names)
    # C = np.zeros((len(cluster_names), len(cluster_names)))

    # calculate entropy from C
    # C_sum = C.values.sum()
    C_sum = C.sum()
    # print("C_sum", C_sum)
    # H = 0
    # # for i in range(len(cluster_names)):
    # #     for j in range(i, len(cluster_names)):
    # #         if (i == j):
    # #             z = C[cluster_names[i]][cluster_names[j]]
    # #         else:
    # #             z = C[cluster_names[i]][cluster_names[j]] + C[cluster_names[j]][cluster_names[i]]
    # #         if z != 0:
    # #             H += -(z/C_sum)*math.log(z/C_sum)
    # for i in range(len(C)):
    #     for j in range(len(C)):
    #         z = C[i, j]
    #         if z != 0:
    #             H += -(z / C_sum) * np.log(z / C_sum)
    # return H
    return _get_spatial_entropy(C, C_sum)

def spatial_coherence_score(adata, annotation_key, degree=4, rep_time=1000, seed=0):
    spatial_coords = adata.obsm['spatial']
    origin_labels = adata.obs[annotation_key].values
    # Use kneighbors_graph to get the adjacency matrix
    neigh = NearestNeighbors(n_neighbors=degree, metric='euclidean').fit(spatial_coords)
    # adjacency_matrix = neigh.kneighbors_graph(n_neighbors=degree, mode='connectivity').toarray().astype(np.int32)
    k_neighbors = neigh.kneighbors(n_neighbors=degree, return_distance=False)
    true_entropy = spatial_entropy(k_neighbors, origin_labels, degree=degree)
    entropies = []
    rng = np.random.default_rng(seed)
    shuffled_labels = origin_labels.copy()
    # for _ in trange(1000):
    for _ in trange(rep_time):
        rng.shuffle(shuffled_labels)
        entropies.append(spatial_entropy(k_neighbors, shuffled_labels, degree=degree))

    return (true_entropy - np.mean(entropies)) / np.std(entropies), true_entropy, entropies


def CHAOS_score(X, pred_labels):
    """
    Calculate the CHAOS score for a given set of spatial coordinates and predicted labels.

    param: X - spatial coordinates
    param: pred_labels - predicted labels

    return: CHAOS score
    """
    from sklearn.preprocessing import StandardScaler
    # Standardize the spatial coordinates
    X = StandardScaler().fit_transform(X)

    # Get the unique cluster labels
    cluster_labels = np.unique(pred_labels)

    # Initialize the distance value and count
    dist_val = 0.
    count = 0

    # Iterate through each cluster
    for k in cluster_labels:
        # Get the spatial coordinates for the current cluster
        cluster_coords = X[pred_labels == k, :]

        # Check if there are at least 2 spatial coordinates in the cluster
        if len(cluster_coords) <= 2:
            continue
        else:
            count += len(cluster_coords)

        # Calculate the distance to the nearest neighbor for each spatial coordinate in the cluster
        nbrs = NearestNeighbors(n_neighbors=1).fit(cluster_coords)
        distances, _ = nbrs.kneighbors()

        # Sum the distances
        dist_val = dist_val + np.sum(distances)

    # Calculate the CHAOS score
    return dist_val / count


def PAS_score(X, pred_labels, k=6):
    """
    Calculate the PAS score for a given set of spatial coordinates and predicted labels.

    param: X - spatial coordinates
    param: pred_labels - predicted labels
    param: k - number of nearest neighbors to consider

    return: PAS score
    """
    # Use NearestNeighbors to find the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    indices = nbrs.kneighbors(return_distance=False)
    # print("type(indices)", type(indices))
    # print("type(pred_labels)", type(pred_labels))
    # print("indices.shape", indices.shape)
    # print("pred_labels.shape", pred_labels.shape)
    # Calculate the PAS score
    return ((pred_labels.reshape(-1, 1) != pred_labels[indices]).sum(1) > k / 2).mean()

from scipy.spatial.distance import squareform,pdist
def ASW_score(X, pred_labels):
    d = squareform(pdist(X))
    return silhouette_score(X=d,labels=pred_labels,metric='precomputed')


def ilisi_score(adata,obs_key="batch_name",clu_key='louvain'):
    ilisi_scores = lisi_graph_py(
        adata=adata,  # 输入的AnnData对象
        obs_key=obs_key,  # 按照'批次'进行分组计算
        n_neighbors=6,  # 计算每个点的邻居数量
        perplexity=None,  # 默认值，LISI函数不使用
        subsample=None,  # 默认不进行数据子采样
        n_cores=1,  # 使用1个核心来并行计算
        verbose=False,  # 是否打印详细信息
    )

    # 计算iLISI分数的中位数，并归一化，结果范围为0到1
    ilisi = np.nanmedian(ilisi_scores)
    print("归一化前ilisi=", ilisi)
    ilisi = (ilisi - 1) / (adata.obs['batch_name'].nunique() - 1)  # 标准化到[0, 1]区间

    print("ilisi=", ilisi)

    print('compute cLISI scores...', flush=True)
    clisi_scores = lisi_graph_py(
        adata=adata,
        obs_key=clu_key,
        n_neighbors=6,
        perplexity=None,
        subsample=None,
        n_cores=1,
        verbose=False,
    )
    clisi = np.nanmedian(clisi_scores)
    print(clisi_scores)
    nlabs = adata.obs[clu_key].nunique()
    print(nlabs)
    clisi = (nlabs - clisi) / (nlabs - 1)
    print("clisi=", clisi)

from sklearn.metrics import adjusted_rand_score as ari_score, normalized_mutual_info_score
import scib
def cal_metric(adata_concat, y_true, y_pred,use_rep, label_key, batch_key="batch_name"):
    ari = ari_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ilisi_scores = scib.me.ilisi_graph(adata_concat, batch_key=batch_key, type_="embed", use_rep=use_rep)
    clisi_scores = scib.me.clisi_graph(adata_concat, label_key=label_key, type_="embed", use_rep=use_rep)
    sb_scores= scib.me.silhouette_batch(adata_concat, batch_key=batch_key, label_key=label_key, embed=use_rep)
    print("ARI = %01.3f, NMI = %01.3f, iLISI=%01.3f, cLISI=%01.3f, SB=%01.3f" %(ari,nmi,ilisi_scores, clisi_scores,sb_scores))
