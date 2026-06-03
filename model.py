import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        # mask矩阵的邻接图 emb矩阵嵌入
        # 矩阵点积 掩码下节点特征总和
        vsum = torch.mm(mask, emb)
        # 按行加和 = 计算每个节点的邻居数
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        # 每一个节点特征/邻居数 = 每个节点特征平均值
        global_emb = vsum / row_sum

        return F.normalize(global_emb, p=2, dim=1)

class Encoder(Module):
    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu, alpha=7):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))

        self.alpha = alpha
        self.conv1X1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.reset_parameters()

        self.disc = Discriminator(self.out_features)
        self.info_nce = InfoNCE()
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        self.adj_dynamic = None

        self.cheb_k = 3

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def info_nce_loss(self, p, p1, p2, temp=0.05):
        loss_ctr = self.info_nce(p, p1, p2, temperature=temp)
        return loss_ctr

    def top_k_binary_mask(self, matrix, k):
        # 找到每行前 k 个最大的数的索引
        _, indices = torch.topk(matrix, k=k, dim=1)

        # 创建掩码矩阵
        mask = torch.zeros_like(matrix, dtype=torch.float32)

        # 使用索引将前 k 个最大的数置为 1
        rows = torch.arange(matrix.size(0)).unsqueeze(1)  # 创建行索引
        mask[rows, indices] = 1

        return mask

    def forward(self, feat, feat_a, adj, adj_f):

        # 空间图
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)

        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)

        emb = self.act(z)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)


        # 特征嵌入 邻接图
        g = self.read(emb, adj)
        g = self.sigm(g)

        g_a = self.read(emb_a, adj)
        g_a = self.sigm(g_a)

        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        # 矩阵更新
        _adj_spatial = adj.unsqueeze(0)  # shape: (1, N, N)
        _adj_feature = adj_f.unsqueeze(0)  # shape: (1, N, N)
        cat_adj = torch.cat((_adj_spatial, _adj_feature), dim=0)##将基因图和图像图进行结合
        cat_adj = self.conv1X1(cat_adj).squeeze(0)##卷积联合
        D = torch.diag(torch.sum(cat_adj, dim=1))
        cat_adj = torch.matmul(torch.inverse(D), cat_adj)
        cat_adj = self.top_k_binary_mask(cat_adj, self.alpha)

        # cat_adj = torch.cat((adj, adj_f), dim=0)
        # cat_adj = self.conv1X1(cat_adj).squeeze(0)
        # D_inv_diag = torch.sum(cat_adj, dim=1)
        # D_inv_diag = 1.0 / D_inv_diag
        # cat_adj = cat_adj * D_inv_diag.unsqueeze(1)
        # cat_adj = self.top_k_binary_mask(cat_adj, self.alpha)

        z_f = F.dropout(feat, 0.0, self.training)
        z_f = torch.mm(z_f, self.weight1)
        z_f = torch.mm(cat_adj, z_f)

        loss = self.info_nce_loss(z,z_f,z_a)

        return hiden_emb, h, ret, ret_a, loss

class InfoNCE(nn.Module):
    def __init__(self, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys, temperature):
        return info_nce(query, positive_key, negative_keys,
                        temperature=temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

def info_nce(query, positive_key, negative_keys=None, temperature=1., reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        # print(logits)
        # mu_p = torch.mean(positive_logit)
        # var_p = torch.var(positive_logit)
        # # print(mu_p, var_p)
        # mu_all = torch.mean(logits)
        # var_all = torch.var(logits)
        # # print(mu_all, var_all)
        # print((mu_p-mu_all)/mu_p)
        # temperature = (var_p.pow(2)-var_all.pow(2))/(-(mu_p-mu_all)+torch.sqrt((mu_p-mu_all).pow(2)+2*(var_p.pow(2)-var_all.pow(2))*np.log(len(positive_logit)*(len(positive_logit)+1)/(2*len(positive_logit)))))
        # print(temperature)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]