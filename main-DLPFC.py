# -*- coding: UTF-8 -*-
import os
import gc
import itertools
import json
import torch

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.linalg
import itertools

from preprocess import Cal_Spatial_Net, preprocess, Cal_Feature_Net
from metric import clustering
from train import train_MVCST
import warnings
warnings.filterwarnings("ignore")
used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# used_device = 'cpu'

from preprocess import  pca
os.environ['R_HOME'] = '/root/miniconda3/envs/SDUCL_CN/lib/R'
os.environ['R_USER'] = '/root/miniconda3/envs/SDUCL_CN/lib/python3.9/site-packages/rpy2'

Batch_list = []
adj_list = []
section_ids = ['151673', '151674', '151675', '151676']
# section_ids = ['151507','151508','151509','151510']
# section_ids = ['151669','151670','151671','151672']


for section_id in section_ids:
    # read data
    print(section_id)
    input_dir = os.path.join('/data/STAGATE_pyG/Data/', section_id)
    #
    # input_dir = os.path.join('/root/autodl-tmp/DLPFC/', section_id)
    adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique(join="++")

    # read the annotation
    Ann_df_layer = pd.read_csv(input_dir + '/metadata.tsv', sep='\t')
    Ann_df = Ann_df_layer['layer_guess']

    adata.obs['Ground Truth'] = Ann_df
    adata = adata[~pd.isnull(adata.obs['Ground Truth'])]

    # make spot name unique
    
    adata.obs_names = [x+'_'+section_id for x in adata.obs_names]
    pca(adata)##使用PCA降维对基因数据进行操作


    # Constructing the spatial network
    Cal_Spatial_Net(adata, rad_cutoff=150) # the spatial network are saved in adata.uns[‘adj’]
    Cal_Feature_Net(adata, k=5)

    # 数据预处理
    if 'highly_variable' not in adata.var.keys():
        adata = preprocess(adata)

    adj_list.append(adata.obsm['adj'])
    Batch_list.append(adata)

adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs['Ground Truth'] = adata_concat.obs['Ground Truth'].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0])
for batch_id in range(1,len(section_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id]))
# adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

iter_comb = list(itertools.combinations(range(len(section_ids)), 2))
adata_concat,trans  = train_MVCST(adata_concat, n_epochs=450, Batch_list=Batch_list, iter_comb=iter_comb, verbose=True, device=used_device, alpha = 3)

clustering(adata_concat, n_clusters=7, key='MVCST',refinement=False)

sc.pp.neighbors(adata_concat, n_neighbors=6)
from metric import cal_metric
cal_metric(adata_concat,adata_concat.obs['Ground Truth'],adata_concat.obs['mclust'],use_rep='emb_pca',label_key='mclust')

sc.pl.spatial(adata_concat, img_key=None, color=['mclust'], title=[''],
                     legend_fontsize=12, show=False, frameon=False,
                      spot_size=150,save="自己的方法(151673-151676)-合并聚类结果可视化.pdf")

from metric import cal_metric

trans_serializable = [t.tolist() if isinstance(t, torch.Tensor) else t for t in trans]
with open('trans(151673-151676).json', 'w') as f:
    json.dump(trans_serializable , f)

# 画图
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (3, 3)
plt.rcParams['font.size'] = 12
plt.rcParams["figure.dpi"] = 300

# Visualization
sc.pp.neighbors(adata_concat, use_rep='MVCST', random_state=666)
sc.tl.umap(adata_concat, random_state=666)

# UMAP图
sc.pl.umap(adata_concat, color=['batch_name', 'Ground Truth', 'mclust'], ncols=3,
           wspace=0.5,show=False, save="自己的方法(151673-151676)-合并批次UMAP-聚类.pdf")
#
# # ARI图
# Batch_list = []
# for section_id in section_ids:
#     Batch_list.append(adata_concat[adata_concat.obs['batch_name'] == section_id])
#
# spot_size = 150
# title_size = 12
# ARI_list = []
# for bb in range(4):
#     ARI_list.append(round(ari_score(Batch_list[bb].obs['Ground Truth'], Batch_list[bb].obs['mclust']), 2))
#
# fig, ax = plt.subplots(1, 4, figsize=(10, 5), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})
# _sc_0 = sc.pl.spatial(Batch_list[0], img_key=None, color=['mclust'], title=[''],
#                       legend_loc=None, legend_fontsize=12, show=False, ax=ax[0], frameon=False,
#                       spot_size=spot_size)
# _sc_0[0].set_title("#"+ section_ids[0], size=title_size)
# _sc_1 = sc.pl.spatial(Batch_list[1], img_key=None, color=['mclust'], title=[''],
#                       legend_loc=None, legend_fontsize=12, show=False, ax=ax[1], frameon=False,
#                       spot_size=spot_size)
# _sc_1[0].set_title("#"+ section_ids[1], size=title_size)
# _sc_2 = sc.pl.spatial(Batch_list[2], img_key=None, color=['mclust'], title=[''],
#                       legend_loc=None, legend_fontsize=12, show=False, ax=ax[2], frameon=False,
#                       spot_size=spot_size)
# _sc_2[0].set_title("#"+ section_ids[2], size=title_size)
# _sc_3 = sc.pl.spatial(Batch_list[3], img_key=None, color=['mclust'], title=[''],
#                       legend_fontsize=12, show=False, ax=ax[3], frameon=False,
#                       spot_size=spot_size)
# _sc_3[0].set_title("#"+ section_ids[3], size=title_size)
# plt.savefig("自己的方法(151673-151676)-单独聚类结果可视化.pdf", dpi=300)
#
# # 151507-151510
# adata_concat.write("(151673-151676)-adata_concat.h5ad")


del adata_concat
del adata
torch.cuda.empty_cache()
gc.collect()