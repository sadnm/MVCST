# -*- coding: UTF-8 -*-
import json
# 人类背外侧前额叶皮层

import os
import torch
import scipy.linalg
import itertools

from preprocess import Cal_Spatial_Net, preprocess, Cal_Feature_Net
from metric import mclust_R, clustering
from train import train_MVCST_Sub
import warnings
warnings.filterwarnings("ignore")
used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = '/root/miniconda3/envs/py310cu118_CN/lib/R'
os.environ['R_USER'] = '/root/miniconda3/envs/py310cu118_CN/lib/python3.9/site-packages/rpy2'
from preprocess import  pca
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import os
# read data
path = 'dataset/Graspot0.1.1/dataset/asp2019/'
Batch_list = []
adj_list = []

section_ids = ['4_5_5','6_5']
# section_ids = ['9','6_5']
for section_id in section_ids:
    # read data
    print(section_id)
    input_dir = os.path.join(path, 'adata_week_' + section_id + '.h5ad')
    adata = sc.read_h5ad(input_dir)
    adata.var_names_make_unique(join="++")

    if 'highly_variable' not in adata.var.keys():
        adata = preprocess(adata)

    adata.obs_names = [x+'_'+section_id for x in adata.obs_names]
    pca(adata)

    Cal_Spatial_Net(adata, rad_cutoff=150) # the spatial network are saved in adata.uns[‘adj’]
    Cal_Feature_Net(adata, k=3)

    adj_list.append(adata.obsm['adj'])
    Batch_list.append(adata)

adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
adj_concat = np.asarray(adj_list[0])
for batch_id in range(1,len(section_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id]))

iter_comb = list(itertools.combinations(range(len(section_ids)), 2))
adata_concat,trans  = train_MVCST_Sub(adata_concat, n_epochs=500, Batch_list=Batch_list, iter_comb=iter_comb, verbose=True, device=used_device, alpha = 7)

trans_serializable = [t.tolist() if isinstance(t, torch.Tensor) else t for t in trans]
with open('trans(4_5_5-6.5).json', 'w') as f:
    json.dump(trans_serializable , f)