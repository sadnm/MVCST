# -*- coding: UTF-8 -*-
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
import scipy.sparse as sp

import torch
used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
from preprocess import Cal_Spatial_Net, preprocess, Cal_Feature_Net
from train import train_MVCST
from preprocess import  pca
## quality control
adata_normal = sc.read_h5ad("dataset/Puck_190921_21_rotation.h5ad")
adata_disease = sc.read_h5ad("dataset/SpatialRNA_cropped_j20_rep1_rotation.h5ad")

print(adata_normal.shape, adata_disease.shape)
sc.pp.filter_cells(adata_normal, min_genes=50)
sc.pp.filter_genes(adata_normal, min_cells=3)
adata_normal.var['mt'] = adata_normal.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
adata_normal = adata_normal[:, ~adata_normal.var['mt']]

sc.pp.filter_genes(adata_disease, min_cells=3)
adata_disease.var['mt'] = adata_disease.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
adata_disease = adata_disease[:, ~adata_disease.var['mt']]

## use common gene between slices
comm_gene = adata_normal.var_names.intersection(adata_disease.var_names)
adata_normal = adata_normal[:, adata_normal.var_names.isin(comm_gene.values)]
adata_disease = adata_disease[:, adata_disease.var_names.isin(comm_gene.values)]

print(adata_normal.shape, adata_disease.shape)

section_ids = ['Puck_190921_21', 'j20_rep1']
Batch_list = []
adj_list = []
for section_id in section_ids:
    if section_id == 'Puck_190921_21':
        adata = adata_normal
        adata.X = sp.csr_matrix(adata.X)
        pca(adata)
        Cal_Spatial_Net(adata, rad_cutoff=45)
        Cal_Feature_Net(adata, k=5)

    else:
        adata = adata_disease
        adata.X = sp.csr_matrix(adata.X)
        pca(adata)
        Cal_Spatial_Net(adata, rad_cutoff=25)
        Cal_Feature_Net(adata, k=5)


    adata.var_names_make_unique(join="++")
    # make spot name unique
    adata.obs_names = [x + '_' + section_id for x in adata.obs_names]

    ## Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]

    adj_list.append(adata.obsm['adj'])
    Batch_list.append(adata)

adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(section_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

adata_concat = train_MVCST(adata_concat, verbose=True, knn_neigh = 100, iter_comb = None, device=used_device)

sc.pp.neighbors(adata_concat, use_rep='Graspot', random_state=666)
sc.tl.umap(adata_concat, random_state=666)
sc.tl.louvain(adata_concat, random_state=666, key_added="louvain", resolution=0.8)

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (3, 3)
_sc_1 = sc.pl.umap(adata_concat, color=['batch_name', 'louvain'], title=[''], ncols=2,
           legend_fontsize=10, wspace=0.9, show=True)

section_ids = ['Puck_190921_21', 'j20_rep1']
for ss in range(len(section_ids)):
    Batch_list[ss].obs['louvain'] = adata_concat[adata_concat.obs['batch_name'] == section_ids[ss]].obs[
        'louvain'].values
    Batch_list[ss].uns['louvain_colors'] = adata_concat.uns['louvain_colors']

import numpy as np

cluster_num = len(adata_concat.obs['louvain'].unique())
cluster_names = np.sort(adata_concat.obs['louvain'].unique())
cluster_names = sorted(cluster_names, key=lambda x: int(x))
import matplotlib.pyplot as plt

spot_size = 1
title_size = 15
fig, ax = plt.subplots(2, cluster_num, figsize=(cluster_num * 3, 7), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

for ss in range(cluster_num):
    _sc_0 = sc.pl.spatial(Batch_list[0], img_key=None, color=['louvain'], title=[''], size=1.5, legend_fontsize=8,
                          show=False, frameon=False, ax=ax[0, ss], spot_size=25, legend_loc=None,
                          groups=[cluster_names[ss]])
    _sc_0[0].set_title('Domain ' + cluster_names[ss], size=title_size)  # 'on data'
    _sc_1 = sc.pl.spatial(Batch_list[-1], img_key=None, color=['louvain'], title=[''], size=1.5, legend_fontsize=8,
                          show=False, frameon=False, ax=ax[1, ss], spot_size=20, legend_loc=None,
                          groups=[cluster_names[ss]])
    _sc_1[0].set_title('Domain ' + cluster_names[ss], size=title_size)

plt.show()