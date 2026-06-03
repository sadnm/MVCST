# -*- coding: UTF-8 -*-
import os
import torch
import pandas as pd
import scanpy as sc
import warnings

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from metric import clustering

warnings.filterwarnings("ignore")
used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
from preprocess import pca, Cal_Spatial_Net, Cal_Feature_Net, preprocess
from train import train_MVCST_Horizontal

os.environ['R_HOME'] = '/root/miniconda3/envs/SDUCL_CN/lib/R'
os.environ['R_USER'] = '/root/miniconda3/envs/SDUCL_CN/lib/python3.9/site-packages/rpy2'

n_clusters = 40
# read data
#file_fold = './Mouse_Brain/' #please replace 'file_fold' with the download path
file_fold = 'dataset/'
adata = sc.read_h5ad(file_fold + 'mouse_anterior_posterior_brain_merged.h5ad')
# adata = sc.read_h5ad(file_fold + 'filtered_feature_bc_matrix.h5ad')
adata.var_names_make_unique()

# print(adata)
pca(adata)
#

# Constructing the spatial network
Cal_Spatial_Net(adata, rad_cutoff=100) # the spatial network are saved in adata.uns[‘adj’]
Cal_Feature_Net(adata, k=3)

# 数据预处理
if 'highly_variable' not in adata.var.keys():
    adata = preprocess(adata)

adata  = train_MVCST_Horizontal(adata, device=used_device)
clustering(adata, n_clusters=n_clusters, key='Graspot',refinement=True)

# plotting spatial clustering result
import matplotlib.pyplot as plt
import seaborn as sns
adata.obsm['spatial'][:,1] = -1*adata.obsm['spatial'][:,1]
rgb_values = sns.color_palette("tab20", len(adata.obs['mclust'].unique()))
color_fine = dict(zip(list(adata.obs['mclust'].unique()), rgb_values))

plt.rcParams["figure.figsize"] = (12, 6)
sc.pl.embedding(adata, basis="spatial",
                color="mclust",
                s=100,
                palette=color_fine,
                show=False,
                save = "Mouse_brain_n=40",
                title='Mouse Anterior & Posterior Brain (Section 1)')
