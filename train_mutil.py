import scipy.sparse as sp
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from model import Encoder
import torch.backends.cudnn as cudnn
# cudnn.deterministic = True
# cudnn.benchmark = True
# import torch.nn.functional as F
from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
from preprocess import concat_adj
import scipy
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def train_MVCST(adata, n_epochs=400, lr=0.001, hidden_dim=64, key_added='MVCST',
                  gradient_clipping=5., weight_decay=0.0001, verbose=False,alpha=3,
                  random_seed=2024, iter_comb=None, Batch_list=None, initial=None, Couple=None,
                  device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')):
    # 设置随机种子
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
   
    section_ids = np.array(adata.obs['batch_name'].unique())

    comm_gene = adata.var_names
    data_list = []

    for adata_tmp in Batch_list:
        adata_tmp = adata_tmp[:, comm_gene]

        adj = adata_tmp.obsm['img_adj']##图像信息

        adj_feature = adata_tmp.obsm['feature_adj']##基因信息
        
        data_list.append(Data(adj=torch.FloatTensor(adj.toarray()),
                            adj_feature = torch.FloatTensor(adj_feature.toarray()),
                            x=torch.FloatTensor(adata_tmp.X.todense()),
                            x_a=torch.FloatTensor(permutation(adata_tmp.X.todense()))))

    model = Encoder(adata.X.shape[1], hidden_dim, alpha = alpha).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if verbose:
        print(model)

    print('Train with MVCST...')

    for epoch in tqdm(range(0, n_epochs)):
        model.train()
        optimizer.zero_grad()
        batch = batch.to(device)
        # 前向传播，计算重构误差
        z, out, ret, ret_a, consistency_loss = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)
        # z, out, ret, ret_a = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)
        # z, out, consistency_loss = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)
        # 重构损失 + OT损失 + 对比损失 + 一致性损失
        loss = + consistency_loss
        # loss = 10 * mse_loss + 1 * ot_loss + consistency_loss

        # print(
        #     f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, mse_loss={mse_loss:.4f}, ot_loss={ot_loss:.4f},cl_loss={cl_loss:.4f},'
        #     # f'consistency_loss={consistency_loss}'
        # )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        with torch.cuda.device('cuda:1'):
            torch.cuda.empty_cache()

##评估
    model.eval()
    with torch.no_grad():
        z_list = []
        for iters, batch in enumerate(data_list):
            batch = batch.to(device)
            z = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)[1].detach().cpu().numpy()
            z_list.append(z)
    adata.obsm[key_added] = np.concatenate(z_list, axis=0)
    print(z_list)