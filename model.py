import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from layer import TransformerBlock
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GCNConv
from utils import get_silhouette_score,vrc_index,get_fast_silhouette
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering, MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from kmedoids import fasterpam
from subgcon import SugbCon
class PretrainModel(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.config = config

        self.Linear1 = nn.Linear(input_dim, self.config.hidden_dim)
        self.encoder = TransformerBlock(hops=config.hops,
                        input_dim=input_dim, 
                        n_layers=config.n_layers,
                        num_heads=config.n_heads,
                        hidden_dim=config.hidden_dim,
                        dropout_rate=config.dropout,
                        attention_dropout_rate=config.attention_dropout) if config.encoder == 'transformer' else SugbCon(input_channels=input_dim,hidden_channels=config.hidden_dim)
        if config.readout == "sum":
            self.readout = global_add_pool
        elif config.readout == "mean":
            self.readout = global_mean_pool
        elif config.readout == "max":
            self.readout = global_max_pool
        else:
            raise ValueError("Invalid pooling type.")
        
        self.marginloss = nn.MarginRankingLoss(0.5)
        self.mlp = MLP(config.hidden_dim*2,config.hidden_dim)
    def forward(self, x):
        node_tensor, neighbor_tensor = self.encoder(x) # (batch_size, 1, hidden_dim), (batch_size, hops, hidden_dim)
        neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.config.device)) # (batch_size, 1, hidden_dim)
        # node_tensor, neighbor_tensor = node_tensor.squeeze(), neighbor_tensor.squeeze()
        # tensor = torch.cat((node_tensor, neighbor_tensor),dim=1)
        # tensor = self.mlp(tensor)
        # return tensor
        return node_tensor.squeeze(), neighbor_tensor.squeeze()

    def contrastive_link_loss(self, node_tensor, neighbor_tensor, adj_, minus_adj):
        

        shuf_index = torch.randperm(node_tensor.shape[0])

        node_tensor_shuf = node_tensor[shuf_index] 
        neighbor_tensor_shuf = neighbor_tensor[shuf_index]

        logits_aa = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor, dim = -1))
        logits_bb = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor_shuf, dim = -1))
        logits_ab = torch.sigmoid(torch.sum(node_tensor * neighbor_tensor_shuf, dim = -1))
        logits_ba = torch.sigmoid(torch.sum(node_tensor_shuf * neighbor_tensor, dim = -1))
        
        TotalLoss = 0.0
        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        TotalLoss += self.marginloss(logits_aa, logits_ba, ones)
        TotalLoss += self.marginloss(logits_bb, logits_ab, ones)
        
        pairwise_similary = torch.mm(node_tensor, node_tensor.t())
        link_loss = minus_adj.multiply(pairwise_similary)-adj_.multiply(pairwise_similary)
        # link_loss = torch.abs(torch.sum(link_loss))/(adj_.shape[0])
        link_loss = torch.sum(link_loss)/(adj_.shape[0]*adj_.shape[0])

        # TotalLoss += 0.001*link_loss
        TotalLoss += self.config.alpha*link_loss

        return TotalLoss

    def cvis_loss(self,tensor,cluster_method,num_clusters,k_init,clustering_metric):
        '''
        :param tensor:
        :param num_clusters: 聚类中心数量（超参数）
        :param k_init: 初始化聚类中心向量次数
        :param clustering_metric: 聚类损失函数类型
        :return:
        '''
        tensor = torch.squeeze(tensor,dim=1)
        norm_tensor = F.normalize(tensor, dim=1)
        cluster_ids_x = self.clustering(tensor.cpu(),cluster_method,num_clusters,k_init)
        if clustering_metric == 'silhouette':
            loss = get_silhouette_score(norm_tensor, cluster_ids_x, goal=1)
        elif clustering_metric == 'fast_silhouette':
            loss = get_fast_silhouette(norm_tensor, cluster_ids_x, goal=1)
        elif clustering_metric == 'vrc':
            loss = vrc_index(norm_tensor, cluster_ids_x)
        return loss
    # method 聚类方法 num_clusters 聚类中心数量 k_init聚类的初始化次数
    def clustering(self,tensor,method,num_clusters,k_init):
        tensor = tensor.detach().numpy()
        if method == 'kmeans':
            km = MiniBatchKMeans(n_clusters=num_clusters, n_init=k_init, init='random')
            cluster_ids_x = torch.tensor(km.fit_predict(tensor))
        elif method == 'kmedoids':
            dists = euclidean_distances(tensor)
            res = fasterpam(dists, medoids=num_clusters, max_iter=500)
            cluster_ids_x = torch.tensor(res.labels.astype('int32'))
        return cluster_ids_x


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden_layers=[128, 64], activation=nn.ReLU()):
        """
        初始化 MLP 网络。
        :param d_in: 输入维度
        :param d_out: 输出维度
        :param hidden_layers: 隐藏层的神经元数量列表，默认为 [128, 64]
        :param activation: 激活函数，默认为 ReLU
        """
        super(MLP, self).__init__()
        layers = []
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(d_in, hidden_layers[0]))
        layers.append(activation)
        # 中间隐藏层
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(activation)
        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_layers[-1], d_out))
        # 构建顺序模块
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数。
        :param x: 输入张量
        :return: 输出张量
        """
        return self.model(x)