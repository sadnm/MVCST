# -*- coding: UTF-8 -*-
import scanpy as sc
import numpy as np
import json
from ST_utils import ICP_align, visualize_2d_alignment_overlay_by_cluster

# read data
path = 'dataset/Graspot0.1.1/dataset/asp2019/'

slice1 = sc.read_h5ad(path+'adata_week_4_5_5.h5ad')
slice2 = sc.read_h5ad(path+'adata_week_6_5.h5ad')
slice3 = sc.read_h5ad(path+'adata_week_9.h5ad')

slice1.obs['clust'] = slice1.obs['res.0.8'].astype(str)
slice2.obs['clust'] = slice2.obs['res.0.8'].astype(str)
slice3.obs['clust'] = slice3.obs['res.0.8'].astype(str)

with open(path + "trans(4-6.5).json", 'r') as file:
    pi1 = json.load(file)
with open(path + "trans(9-6.5).json", 'r') as file:
    pi2 = json.load(file)

slices = [slice1, slice2, slice3]
pi1 = np.array(pi1[0])
pi2 = np.array(pi2[0])
pi1 = pi1 / (pi1.sum(axis=1, keepdims=True) + 1e-8)
pi2 = pi2 / (pi2.sum(axis=1, keepdims=True) + 1e-8)

ref_index = 1
aligned_slices = [None] * 3
aligned_slices[ref_index] = slices[ref_index].copy()  # 中间切片保持不动

# 对齐 adata_0 到 adata_1
print("Aligning adata_0 → adata_1")
coords_0 = ICP_align(
    adata_target=slices[0],
    adata_ref=slices[1],
    prob_matrix=pi1,
    plot_align=False
)
slices[0].obsm['spatial'] = coords_0


print("Aligning adata_2 → adata_1")
coords_2 = ICP_align(
    adata_target=slices[2],
    adata_ref=slices[1],
    prob_matrix=pi2,  # shape: (N2, N1)，和 adata_2 → adata_1 匹配
    plot_align=False
)

slices[2].obsm['spatial'] = coords_2

print(coords_0)
print("=======================================")
print(coords_2)
print("=======================================")
print(slice2.obsm['spatial'])

visualize_2d_alignment_overlay_by_cluster(slice2,slice3)