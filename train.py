import scipy.sparse as sp
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from model import Encoder

import torch.backends.cudnn as cudnn
from OT_utils import unbalanced_ot, unbalanced_ot_parameter
from ST_utils import permutation, add_contrastive_label

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from preprocess import concat_adj
import scipy
import ot


def norm_and_center_coordinates(X):
    """
    Normalizes and centers coordinates at the origin.

    Args:
        X: Numpy array

    Returns:
        X_new: Updated coordiantes.
    """
    return (X - X.mean(axis=0)) / min(scipy.spatial.distance.pdist(X))


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

    # 获取所有切片的section_id
    section_ids = np.array(adata.obs['batch_name'].unique())

    comm_gene = adata.var_names
    data_list = []

    for adata_tmp in Batch_list:
        adata_tmp = adata_tmp[:, comm_gene]

        adj = adata_tmp.obsm['adj']

        adj_feature = adata_tmp.obsm['feature_adj']
        
        data_list.append(Data(adj=torch.FloatTensor(adj.toarray()),
                              adj_feature = torch.FloatTensor(adj_feature.toarray()),
                              x=torch.FloatTensor(adata_tmp.X.todense()),
                              x_a=torch.FloatTensor(permutation(adata_tmp.X.todense()))))

    model = Encoder(adata.X.shape[1], hidden_dim, alpha = alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if verbose:
        print(model)

    print('Train with MVCST...')
    pair_data_list = []
    for comb in iter_comb:
        # print(comb)
        i, j = comb[0], comb[1]
        # 从原始adata选择这两个batch数据
        batch_pair = adata[adata.obs['batch_name'].isin([section_ids[i], section_ids[j]])]

        label_CSL = add_contrastive_label(batch_pair)

        adj_1 = Batch_list[i].obsm['adj']
        adj_2 = Batch_list[j].obsm['adj']

        adj_combined = concat_adj(adj_1,adj_2)

        adj_feature_1 = Batch_list[i].obsm['feature_adj']
        adj_feature_2 = Batch_list[j].obsm['feature_adj']

        adj_feature_combined = concat_adj(adj_feature_1,adj_feature_2)

        pair_data_list.append(Data(adj=torch.FloatTensor(adj_combined.toarray()),
                                   adj_feature=torch.FloatTensor(adj_feature_combined.toarray()),
                                   # edge_index=torch.LongTensor(np.array([edge_pairs[0], edge_pairs[1]])),
                                   label_CSL=torch.FloatTensor(label_CSL),
                                   x=torch.FloatTensor(batch_pair.X.todense()),
                                   x_a=torch.FloatTensor(permutation(batch_pair.X).todense())))

    pair_loader = DataLoader(pair_data_list, batch_size=1, shuffle=False)

    tran_list = []
    for iters, batch in enumerate(pair_loader):
        if initial == True:
            # 传输矩阵初始化
            ax = Batch_list[iter_comb[iters][0]].obsm['spatial']
            ay = Batch_list[iter_comb[iters][1]].obsm['spatial']
            dist = scipy.spatial.distance_matrix(norm_and_center_coordinates(ax), norm_and_center_coordinates(ay))
            # dist = scipy.spatial.distance_matrix(ax,ay)
            n1 = ax.shape[0]
            n2 = ay.shape[0]
            pi0 = pi = ot.sinkhorn(np.ones(n1) / n1, np.ones(n2) / n2, dist, reg=0.02)
            tran = torch.tensor(pi0, dtype=torch.float).to(device)
        else:
            tran = None

        num = []
        for i in [Batch_list[iter_comb[iters][0]], Batch_list[iter_comb[iters][1]]]:
            num.append(i.X.shape[0])
        loss_CSL = nn.BCEWithLogitsLoss()

        # 进行训练
        for epoch in tqdm(range(0, n_epochs)):
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)

            # 前向传播，计算重构误差
            z, out, ret, ret_a, consistency_loss = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)
            # z, out, ret, ret_a = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)
            # z, out, consistency_loss = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)


            loss_sl_1 = loss_CSL(ret, batch.label_CSL)
            loss_sl_2 = loss_CSL(ret_a, batch.label_CSL)##全局局部互信息对比学习
            cl_loss = loss_sl_1 + loss_sl_2
            mse_loss = F.mse_loss(batch.x, out)##通过编码器

            ds1 = z[0:num[0], :]  # 第一个batch嵌入表示
            ds2 = z[num[0]:num[0] + num[1], :]  # 第二个batch嵌入表示

            # 最优传输损失
            ot_loss, tran = unbalanced_ot(tran, ds1, ds2, device=device, Couple=Couple, reg=0.1, reg_m=1.0)

            # 重构损失 + OT损失 + 对比损失 + 一致性损失
            loss = 10 * mse_loss + 1 * ot_loss + 2 * cl_loss + consistency_loss
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

        # torch.cuda.empty_cache()
        tran_list.append(tran)

    #
    model.eval()
    with torch.no_grad():
        z_list = []
        for iters, batch in enumerate(data_list):
            batch = batch.to(device)
            z = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)[1].detach().cpu().numpy()
            z_list.append(z)
    adata.obsm[key_added] = np.concatenate(z_list, axis=0)
    print(z_list)

    # for comb in iter_comb:
    #     print(comb)
    #     i, j = comb[0], comb[1]
    #     z_sublist = [z_list[i],z_list[j]]
    #     batch_a = np.zeros(len(z_list[i]))
    #     batch_b = np.ones(len(z_list[j]))
    #     batch = np.concatenate([batch_a, batch_b])
    #     # adata.obs['batch'] = batch
    #     adata.obsm['X_emb'] = np.concatenate(z_sublist)

    return adata, tran_list


def train_MVCST_Sub(adata, n_epochs=400, lr=0.001, hidden_dim=64, key_added='MVCST',
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

    # 获取所有切片的section_id
    section_ids = np.array(adata.obs['batch_name'].unique())

    comm_gene = adata.var_names
    data_list = []

    for adata_tmp in Batch_list:
        adata_tmp = adata_tmp[:, comm_gene]
        adj = adata_tmp.obsm['adj']
        adj_feature = adata_tmp.obsm['feature_adj']
        data_list.append(Data(adj=torch.FloatTensor(adj.toarray()),
                              adj_feature = torch.FloatTensor(adj_feature.toarray()),
                              x=torch.FloatTensor(adata_tmp.X),
                              x_a=torch.FloatTensor(permutation(adata_tmp.X))))

    model = Encoder(adata.X.shape[1], hidden_dim, alpha = alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if verbose:
        print(model)

    print('Train with MVCST...')
    pair_data_list = []
    for comb in iter_comb:
        # print(comb)
        i, j = comb[0], comb[1]
        # 从原始adata选择这两个batch数据
        batch_pair = adata[adata.obs['batch_name'].isin([section_ids[i], section_ids[j]])]

        label_CSL = add_contrastive_label(batch_pair)

        adj_1 = Batch_list[i].obsm['adj']
        adj_2 = Batch_list[j].obsm['adj']

        adj_combined = concat_adj(adj_1,adj_2)

        adj_feature_1 = Batch_list[i].obsm['feature_adj']
        adj_feature_2 = Batch_list[j].obsm['feature_adj']

        adj_feature_combined = concat_adj(adj_feature_1,adj_feature_2)

        pair_data_list.append(Data(adj=torch.FloatTensor(adj_combined.toarray()),
                                   adj_feature=torch.FloatTensor(adj_feature_combined.toarray()),
                                   # edge_index=torch.LongTensor(np.array([edge_pairs[0], edge_pairs[1]])),
                                   label_CSL=torch.FloatTensor(label_CSL),
                                   x=torch.FloatTensor(batch_pair.X),
                                   x_a=torch.FloatTensor(permutation(batch_pair.X))))

    pair_loader = DataLoader(pair_data_list, batch_size=1, shuffle=False)

    tran_list = []
    for iters, batch in enumerate(pair_loader):
        if initial == True:
            # 传输矩阵初始化
            ax = Batch_list[iter_comb[iters][0]].obsm['spatial']
            ay = Batch_list[iter_comb[iters][1]].obsm['spatial']
            dist = scipy.spatial.distance_matrix(norm_and_center_coordinates(ax), norm_and_center_coordinates(ay))
            # dist = scipy.spatial.distance_matrix(ax,ay)
            n1 = ax.shape[0]
            n2 = ay.shape[0]
            pi0 = pi = ot.sinkhorn(np.ones(n1) / n1, np.ones(n2) / n2, dist, reg=0.02)
            tran = torch.tensor(pi0, dtype=torch.float).to(device)
        else:
            tran = None

        num = []
        for i in [Batch_list[iter_comb[iters][0]], Batch_list[iter_comb[iters][1]]]:
            num.append(i.X.shape[0])
        loss_CSL = nn.BCEWithLogitsLoss()

        # 进行训练
        for epoch in tqdm(range(0, n_epochs)):
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)

            # 前向传播，计算重构误差
            z, out, ret, ret_a, consistency_loss = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)
            # z, out, ret, ret_a = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)
            # z, out, consistency_loss = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)


            loss_sl_1 = loss_CSL(ret, batch.label_CSL)
            loss_sl_2 = loss_CSL(ret_a, batch.label_CSL)
            cl_loss = loss_sl_1 + loss_sl_2
            mse_loss = F.mse_loss(batch.x, out)

            ds1 = z[0:num[0], :]  # 第一个batch嵌入表示
            ds2 = z[num[0]:num[0] + num[1], :]  # 第二个batch嵌入表示

            # 最优传输损失
            ot_loss, tran = unbalanced_ot(tran, ds1, ds2, device=device, Couple=Couple, reg=0.1, reg_m=1.0)

            # 重构损失 + OT损失 + 对比损失 + 一致性损失
            loss = 10 * mse_loss + 1 * ot_loss + 2 * cl_loss + consistency_loss
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

        # torch.cuda.empty_cache()
        tran_list.append(tran)

    #
    model.eval()
    with torch.no_grad():
        z_list = []
        for iters, batch in enumerate(data_list):
            batch = batch.to(device)
            z = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)[1].detach().cpu().numpy()
            z_list.append(z)
    adata.obsm[key_added] = np.concatenate(z_list, axis=0)
    print(z_list)

    # for comb in iter_comb:
    #     print(comb)
    #     i, j = comb[0], comb[1]
    #     z_sublist = [z_list[i],z_list[j]]
    #     batch_a = np.zeros(len(z_list[i]))
    #     batch_b = np.ones(len(z_list[j]))
    #     batch = np.concatenate([batch_a, batch_b])
    #     # adata.obs['batch'] = batch
    #     adata.obsm['X_emb'] = np.concatenate(z_sublist)

    return adata, tran_list

def train_MVCST_Horizontal(adata, n_epochs=1000, lr=0.001, hidden_dim=64, key_added='MVCST',
                  gradient_clipping=5., weight_decay=0.0001,alpha=3,
                  random_seed=2024,
                  device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')):
    # 设置随机种子
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


    adj = adata.obsm['adj']
    adj_feature = adata.obsm['feature_adj']
    label_CSL = add_contrastive_label(adata)
    batch = Data(adj=torch.FloatTensor(adj.toarray()),
         adj_feature=torch.FloatTensor(adj_feature.toarray()),
         # edge_index=torch.LongTensor(np.array([edge_pairs[0], edge_pairs[1]])),
         label_CSL=torch.FloatTensor(label_CSL),
         x=torch.FloatTensor(adata.X.todense()),
         x_a=torch.FloatTensor(permutation(adata.X).todense()))

    model = Encoder(adata.X.shape[1], hidden_dim, alpha=alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print('Train with MVCST...')
    loss_CSL = nn.BCEWithLogitsLoss()

    # 进行训练
    for epoch in tqdm(range(0, n_epochs)):
        model.train()
        optimizer.zero_grad()
        batch = batch.to(device)

        # 前向传播，计算重构误差
        z, out, ret, ret_a, consistency_loss = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)

        loss_sl_1 = loss_CSL(ret, batch.label_CSL)
        loss_sl_2 = loss_CSL(ret_a, batch.label_CSL)
        cl_loss = loss_sl_1 + loss_sl_2
        mse_loss = F.mse_loss(batch.x, out)

        # 重构损失 + OT损失 + 对比损失 + 一致性损失



        loss = 10 * mse_loss + 1 * cl_loss + consistency_loss

        print(
            f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, mse_loss={mse_loss:.4f} ,cl_loss={cl_loss:.4f},'
            f'consistency_loss={consistency_loss}'
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    #     torch.cuda.empty_cache()
    # torch.cuda.empty_cache()

    #
    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        z = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)[1].detach().cpu().numpy()
    adata.obsm[key_added] = z


    return adata


def train_MVCST_ablation(adata, n_epochs=400, lr=0.001, hidden_dim=64, key_added='MVCST',
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

    # 获取所有切片的section_id
    section_ids = np.array(adata.obs['batch_name'].unique())

    comm_gene = adata.var_names
    data_list = []

    for adata_tmp in Batch_list:
        adata_tmp = adata_tmp[:, comm_gene]
        adj = adata_tmp.obsm['adj']
        adj_feature = adata_tmp.obsm['feature_adj']
        data_list.append(Data(adj=torch.FloatTensor(adj.toarray()),
                              adj_feature = torch.FloatTensor(adj_feature.toarray()),
                              x=torch.FloatTensor(adata_tmp.X.todense()),
                              x_a=torch.FloatTensor(permutation(adata_tmp.X.todense()))))

    model = Encoder(adata.X.shape[1], hidden_dim, alpha = alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if verbose:
        print(model)

    print('Train with MVCST...')
    pair_data_list = []
    for comb in iter_comb:
        # print(comb)
        i, j = comb[0], comb[1]
        # 从原始adata选择这两个batch数据
        batch_pair = adata[adata.obs['batch_name'].isin([section_ids[i], section_ids[j]])]

        label_CSL = add_contrastive_label(batch_pair)

        adj_1 = Batch_list[i].obsm['adj']
        adj_2 = Batch_list[j].obsm['adj']

        adj_combined = concat_adj(adj_1,adj_2)

        adj_feature_1 = Batch_list[i].obsm['feature_adj']
        adj_feature_2 = Batch_list[j].obsm['feature_adj']

        adj_feature_combined = concat_adj(adj_feature_1,adj_feature_2)

        pair_data_list.append(Data(adj=torch.FloatTensor(adj_combined.toarray()),
                                   adj_feature=torch.FloatTensor(adj_feature_combined.toarray()),
                                   # edge_index=torch.LongTensor(np.array([edge_pairs[0], edge_pairs[1]])),
                                   label_CSL=torch.FloatTensor(label_CSL),
                                   x=torch.FloatTensor(batch_pair.X.todense()),
                                   x_a=torch.FloatTensor(permutation(batch_pair.X).todense())))

    pair_loader = DataLoader(pair_data_list, batch_size=1, shuffle=False)

    tran_list = []
    for iters, batch in enumerate(pair_loader):
        if initial == True:
            # 传输矩阵初始化
            ax = Batch_list[iter_comb[iters][0]].obsm['spatial']
            ay = Batch_list[iter_comb[iters][1]].obsm['spatial']
            dist = scipy.spatial.distance_matrix(norm_and_center_coordinates(ax), norm_and_center_coordinates(ay))
            # dist = scipy.spatial.distance_matrix(ax,ay)
            n1 = ax.shape[0]
            n2 = ay.shape[0]
            pi0 = pi = ot.sinkhorn(np.ones(n1) / n1, np.ones(n2) / n2, dist, reg=0.02)
            tran = torch.tensor(pi0, dtype=torch.float).to(device)
        else:
            tran = None

        num = []
        for i in [Batch_list[iter_comb[iters][0]], Batch_list[iter_comb[iters][1]]]:
            num.append(i.X.shape[0])
        loss_CSL = nn.BCEWithLogitsLoss()

        # 进行训练
        for epoch in tqdm(range(0, n_epochs)):
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)

            # 前向传播，计算重构误差
            z, out, ret, ret_a, consistency_loss = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)
            # z, out, ret, ret_a = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)
            # z, out, consistency_loss = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)


            loss_sl_1 = loss_CSL(ret, batch.label_CSL)
            loss_sl_2 = loss_CSL(ret_a, batch.label_CSL)
            cl_loss = loss_sl_1 + loss_sl_2
            mse_loss = F.mse_loss(batch.x, out)

            ds1 = z[0:num[0], :]  # 第一个batch嵌入表示
            ds2 = z[num[0]:num[0] + num[1], :]  # 第二个batch嵌入表示

            # 最优传输损失
            ot_loss, tran = unbalanced_ot(tran, ds1, ds2, device=device, Couple=Couple, reg=0.1, reg_m=1.0)

            # 重构损失 + OT损失 + 对比损失 + 一致性损失
            # loss = 10 * mse_loss + 1 * ot_loss + 1 * cl_loss + consistency_loss
            # loss = 10 * mse_loss + 1 * ot_loss + consistency_loss
            # loss = 10 * mse_loss + 1 * cl_loss + consistency_loss
            loss = 10 * mse_loss + 1 * ot_loss + 1 * cl_loss


            # print(
            #     f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, mse_loss={mse_loss:.4f}, ot_loss={ot_loss:.4f},cl_loss={cl_loss:.4f},'
            #     # f'consistency_loss={consistency_loss}'
            # )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
        with torch.cuda.device('cuda:1'):
            torch.cuda.empty_cache()

        # torch.cuda.empty_cache()
        tran_list.append(tran)

    #
    model.eval()
    with torch.no_grad():
        z_list = []
        for iters, batch in enumerate(data_list):
            batch = batch.to(device)
            z = model(batch.x, batch.x_a, batch.adj, batch.adj_feature)[1].detach().cpu().numpy()
            z_list.append(z)
    adata.obsm[key_added] = np.concatenate(z_list, axis=0)
    print(z_list)

    # for comb in iter_comb:
    #     print(comb)
    #     i, j = comb[0], comb[1]
    #     z_sublist = [z_list[i],z_list[j]]
    #     batch_a = np.zeros(len(z_list[i]))
    #     batch_b = np.ones(len(z_list[j]))
    #     batch = np.concatenate([batch_a, batch_b])
    #     # adata.obs['batch'] = batch
    #     adata.obsm['X_emb'] = np.concatenate(z_sublist)

    return adata, tran_list
