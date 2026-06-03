# -*- coding: UTF-8 -*-
import itertools
import warnings

warnings.filterwarnings("ignore")

import os

os.environ['R_HOME'] = '/root/miniconda3/envs/SDUCL_CN/lib/R'
os.environ['R_USER'] = '/root/miniconda3/envs/SDUCL_CN/lib/python3.9/site-packages/rpy2'

import torch
import scanpy as sc
from train import train_MVCST

used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

section_ids = ['Stereo-seq', 'Slide-seqV2']

adata_concat = sc.read('dataset/adata_concat_MoB.h5ad')
# train
iter_comb = list(itertools.combinations(range(len(section_ids)), 2))
adata_concat, _ = train_MVCST(adata_concat, n_epochs=200, Batch_list=None, iter_comb=iter_comb,
                                device=used_device, alpha=3)  # epochs = 1500,

# adata_concat.write('adata_concat_MoB.h5ad')
# Clustering
sc.pp.neighbors(adata_concat, use_rep='Graspot', random_state=666)
sc.tl.louvain(adata_concat, random_state=666, key_added="louvain", resolution=0.5)

# Visualization
sc.tl.umap(adata_concat, random_state=666)

section_color = ['#ff7f0e', '#1f77b4']
section_color_dict = dict(zip(section_ids, section_color))
adata_concat.uns['batch_name_colors'] = [section_color_dict[x] for x in adata_concat.obs.batch_name.cat.categories]

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (3, 3)
plt.rcParams['font.size'] = 10

sc.pl.umap(adata_concat, color=['batch_name', 'louvain'], ncols=2, wspace=0.5, show=False,
           save='umap-MoB-自己的方法.pdf')
sc.pl.umap(adata_concat, color=['batch_name'],
           wspace=0.5, show=False, legend_loc='none',
           save="自己的方法-MoB-合并批次UMAP.pdf")

sc.pl.umap(adata_concat, color=['louvain'],
           wspace=0.5, show=False, legend_loc='on data', legend_fontoutline=2,
           save="自己的方法-MoB-合并批次聚类.pdf")

import matplotlib.pyplot as plt

spot_size = 50
title_size = 15
fig, ax = plt.subplots(2, 1, figsize=(6, 9), gridspec_kw={'wspace': 0.05, 'hspace': 0.2})
_sc_0 = sc.pl.spatial(adata_concat[adata_concat.obs['batch_name'] == 'Slide-seqV2'], img_key=None, color=['louvain'],
                      title=['Slide-seqV2'],
                      legend_fontsize=10, show=False, ax=ax[0], frameon=False, spot_size=spot_size, legend_loc=None)
_sc_0[0].set_title('Slide-seqV2', size=title_size)

_sc_1 = sc.pl.spatial(adata_concat[adata_concat.obs['batch_name'] == 'Stereo-seq'], img_key=None, color=['louvain'],
                      title=['Stereo-seq'],
                      legend_fontsize=10, show=False, ax=ax[1], frameon=False, spot_size=spot_size)
_sc_1[0].set_title('Stereo-seq', size=title_size)
_sc_1[0].invert_yaxis()
plt.savefig('自己的方法-MoB-聚类.pdf')

import numpy as np

cluster_num = len(adata_concat.obs['louvain'].unique())
cluster_names = np.sort(adata_concat.obs['louvain'].unique())
cluster_names = sorted(cluster_names, key=lambda x: int(x))
import matplotlib.pyplot as plt

spot_size = 1
title_size = 15
fig, ax = plt.subplots(2, cluster_num, figsize=(cluster_num * 3, 7), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})

for ss in range(cluster_num):
    _sc_0 = sc.pl.spatial(adata_concat[adata_concat.obs['batch_name'] == 'Slide-seqV2'], img_key=None,
                          color=['louvain'], title=[''], size=1.5, legend_fontsize=8,
                          show=False, frameon=False, ax=ax[0, ss], spot_size=25, legend_loc=None,
                          groups=[cluster_names[ss]])
    _sc_0[0].set_title('Domain ' + cluster_names[ss], size=title_size)  # 'on data'
    _sc_1 = sc.pl.spatial(adata_concat[adata_concat.obs['batch_name'] == 'Stereo-seq'], img_key=None, color=['louvain'],
                          title=[''], size=1.5, legend_fontsize=8,
                          show=False, frameon=False, ax=ax[1, ss], spot_size=20, legend_loc=None,
                          groups=[cluster_names[ss]])
    _sc_1[0].set_title('Domain ' + cluster_names[ss], size=title_size)

plt.savefig('自己的方法-MoB-聚类-不同域.pdf')

from metric import ilisi_score
ilisi_score(adata_concat)