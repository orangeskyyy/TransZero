import argparse
import torch
import scipy.sparse as sp
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import dgl
from sklearn.metrics import f1_score
import scipy.sparse as sp
import numpy as np
import networkx as nx
from numpy import *
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score

# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Choose from {pubmed}')
    parser.add_argument('--device', type=int, default=0,
                        help='Device cuda id')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed.')

    # model parameters
    parser.add_argument('--hops', type=int, default=7,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--pe_dim', type=int, default=15,
                        help='position embedding size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    parser.add_argument('--readout', type=str, default="mean")
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help='the value the balance the loss.')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    parser.add_argument('--group_epoch_gap', type=int, default=20,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=50, 
                        help='Patience for early stopping')
    
    # model saving
    parser.add_argument('--save_path', type=str, default='./model/',
                        help='The path for the model to save')
    parser.add_argument('--model_name', type=str, default='cora',
                        help='The name for the model to save')
    parser.add_argument('--embedding_path', type=str, default='./pretrain_result/',
                        help='The path for the embedding to save')

    # model ablation
    parser.add_argument('--encoder',type=str,default='gcn')
    parser.add_argument('--loss',type=str,default='cl',help='损失函数消融')
    parser.add_argument('--cluster',type=str,default='kmedoids')

    # clustering parameters
    parser.add_argument('--num_clusters', type=int, default=5,help='聚类中心数量')
    parser.add_argument('--k_init', type=int, default=5, help='聚类初始化次数')
    parser.add_argument('--cvi_method', type=str, default="silhouette",help='cvi指数计算方法')
    return parser.parse_args()


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    #adjacency_matrix(transpose, scipy_fmt="csr")
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = g.adj_external(scipy_fmt='csr')
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    return lap_pos_enc


'''
adj 邻接矩阵
features 节点特征
K 子图生成最大邻居条数（本文最大设置为5）
'''
def re_features(adj, features, K):
    #传播之后的特征矩阵,size= (N, 1, K+1, d )
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])

    for i in range(features.shape[0]):

        nodes_features[i, 0, 0, :] = features[i]
    # x是features的深拷贝
    x = features + torch.zeros_like(features)
    
    for i in range(K):
        # 通过邻接矩阵 adj 对特征 x 进行矩阵乘法可以进行特征传播
        # N*d
        x = torch.matmul(adj, x)

        for index in range(features.shape[0]):

            nodes_features[index, 0, i + 1, :] = x[index]        
    # 使用 squeeze() 方法去除维度为 1 的维度，使最终的 nodes_features 形状为 (N, K+1, d)，表示每个节点在每个传播步骤的特征。
    nodes_features = nodes_features.squeeze()


    return nodes_features

def conductance_hop(adj, max_khop):
    # 确保邻接矩阵的数据类型为 torch.float
    adj = adj.to(dtype=torch.float)
    # 初始化当前跳数的邻接矩阵为原始邻接矩阵
    adj_current_hop = adj

    # 创建一个全零的张量 results，用于存储每一跳的计算结果
    # 形状为 (max_khop + 1, adj.shape[0])，其中 max_khop + 1 表示跳数（包括 0 跳），adj.shape[0] 表示节点数
    results = torch.zeros((max_khop+1, adj.shape[0]))

    # 遍历从 0 到 max_khop 的每一跳
    for hop in range(max_khop+1):
        # 计算当前跳数的邻接矩阵与原始邻接矩阵的矩阵乘法，得到下一跳的邻接矩阵
        adj_current_hop = torch.matmul(adj_current_hop, adj)
        # 计算每一列的元素和，即每个节点的度（下一跳的度）
        degree = torch.sum(adj_current_hop, dim=0)
        # 对当前跳数的邻接矩阵取符号，将非零元素变为 1，零元素保持为 0
        adj_current_hop_sign = torch.sign(adj_current_hop)
        # 计算符号矩阵每一列的元素和，即每个节点的非零邻居数量
        degree_1 = torch.sum(adj_current_hop_sign, dim=0)
        # 将 (degree - degree_1) 的结果转换为密集张量，并调整形状为 (1, -1) 后存储到 results 的第 hop 行
        results[hop] = (degree-degree_1).to_dense().reshape(1, -1)

    # 对 results 进行转置，将形状变为 (adj.shape[0], max_khop + 1)
    results = results.T
    # 找到每一行的最大值所在的索引
    max_indices = torch.argmax(results, dim=1)

    # 遍历 results 的每一行
    for i in range(results.shape[0]):
        # 遍历 results 的每一列
        for j in range(results.shape[1]):
            # 如果当前列索引 j 大于该行最大值所在的索引，并且最大值所在的索引不为 0
            if j>max_indices[i] and max_indices[i] != 0:
                # 将该位置的元素置为 0
                results[i][j] = 0
            else:
                # 否则将该位置的元素置为 1
                results[i][j] = 1

    if max_khop == 0:
        results = torch.ones((adj.shape[0], max_khop + 1))
    # 返回最终的结果矩阵
    return results

# def f1_score_calculation(y_pred, y_true):
#     if len(y_pred.shape) == 1:
#         y_pred = y_pred.reshape(1, -1)
#         y_true = y_true.reshape(1, -1)
#     F1 = []
#     for i in range(y_pred.shape[0]):
#         pre = torch.sum(torch.multiply(y_pred[i], y_true[i]))/(torch.sum(y_pred[i])+1E-9)
#         rec = torch.sum(torch.multiply(y_pred[i], y_true[i]))/(torch.sum(y_true[i])+1E-9)
#         F1.append(2 * pre * rec / (pre + rec+1E-9))

#     return mean(F1)




def f1_score_calculation(y_pred, y_true,args=None):
    y_pred = y_pred.reshape(1, -1)
    y_true = y_true.reshape(1, -1)
    pre = torch.sum(torch.multiply(y_pred, y_true))/(torch.sum(y_pred)+1E-9)
    rec = torch.sum(torch.multiply(y_pred, y_true))/(torch.sum(y_true)+1E-9)
    F1 = 2 * pre * rec / (pre + rec+1E-9)
    return F1


def evaluation(comm_find, comm):
    
    comm_find = comm_find.reshape(-1)
    comm = comm.reshape(-1)
    
    return normalized_mutual_info_score(comm, comm_find), adjusted_rand_score(comm, comm_find), jaccard_score(comm, comm_find)

# def evaluation(comm_find, comm):

#     nmi_all, ari_all, jac_all = [], [], []

#     for i in range(comm_find.shape[0]):
#         nmi_all.append(NMI_score(comm_find[i], comm[i]))
#         ari_all.append(ARI_score(comm_find[i], comm[i]))
#         jac_all.append(JAC_score(comm_find[i], comm[i]))

#     return np.mean(nmi_all), np.mean(ari_all), np.mean(jac_all) 

def NMI_score(comm_find, comm):

    score = normalized_mutual_info_score(comm, comm_find)
    #print("q, nmi:", score)
    return score

def ARI_score(comm_find, comm):

    score = adjusted_rand_score(comm, comm_find)
    #print("q, ari:", score)

    return score

def JAC_score(comm_find, comm):

    score = jaccard_score(comm, comm_find)
    #print("q, jac:", score)
    return score

def load_query_n_gt(path, dataset, vec_length):
    # load query and ground truth
    query = []
    file_query = open(path + dataset + '/' + dataset + ".query", 'r')
    for line in file_query:
        vec = [0 for i in range(vec_length)]
        line = line.strip()
        line = line.split(" ")
        for i in line:
            vec[int(i)] = 1
        query.append(vec)

    gt = []
    file_gt = open(path + dataset + '/' + dataset + ".gt", 'r')
    for line in file_gt:
        vec = [0 for i in range(vec_length)]
        line = line.strip()
        line = line.split(" ")
        
        for i in line:
            vec[int(i)] = 1
        gt.append(vec)
    
    return torch.Tensor(query), torch.Tensor(gt)

def get_gt_legnth(path, dataset):
    gt_legnth = []
    file_gt = open(path + dataset + '/' + dataset + ".gt", 'r')
    for line in file_gt:
        line = line.strip()
        line = line.split(" ")
        gt_legnth.append(len(line))
    
    return torch.Tensor(gt_legnth)

def cosin_similarity(query_tensor, emb_tensor):
    # similarity = torch.stack([torch.cosine_similarity(query_tensor[i], emb_tensor, dim=1) for i in range(len(query_tensor))], 0)
    similarity = torch.stack([torch.cosine_similarity(query_tensor[i].reshape(1, -1), emb_tensor, dim=1) for i in range(len(query_tensor))], 0)
    # print(similarity.shape)
    return similarity
    
def dot_similarity(query_tensor, emb_tensor):
    similarity = torch.mm(query_tensor, emb_tensor.t()) # (query_num, node_num)
    similarity = torch.nn.Softmax(dim=1)(similarity)
    return similarity


def transform_coo_to_csr(adj):
    row=adj._indices()[0]
    col=adj._indices()[1]
    data=adj._values()
    shape=adj.size()
    adj=sp.csr_matrix((data, (row, col)), shape=shape)
    return adj

def transform_csr_to_coo(adj, size=None):
    adj = adj.tocoo()
    adj = torch.sparse.LongTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                              torch.LongTensor(adj.data.astype(np.int32)),
                              torch.Size([size, size]))
    return adj

def transform_sp_csr_to_coo(adj, batch_size, node_num):
    # chunks
    node_index = [i for i in range(node_num)]
    divide_index = [node_index[i:i+batch_size] for i in range(0, len(node_index), batch_size)]

    # adj of each chunks, in the format of sp_csr
    print("start mini batch: adj of each chunks")
    adj_sp_csr = [adj[divide_index[i]][:, divide_index[i]] for i in range(len(divide_index))]
    print("start mini batch: minus adj of each chunks")
    minus_adj_sp_csr = [sp.csr_matrix(torch.ones(item.shape))-item for item in adj_sp_csr]

    # adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in adj_sp_csr]
    # minus_adj_tensor_coo = [transform_csr_to_coo(item).to_dense() for item in minus_adj_sp_csr]
    print("start mini batch: back to torch coo adj")
    adj_tensor_coo = [transform_csr_to_coo(adj_sp_csr[i], len(divide_index[i])).to_dense() for i in range(len(divide_index))]
    print("start mini batch: back to torch coo minus adj")
    minus_adj_tensor_coo = [transform_csr_to_coo(minus_adj_sp_csr[i], len(divide_index[i])).to_dense() for i in range(len(divide_index))]

    return adj_tensor_coo, minus_adj_tensor_coo


# transform coo to edge index in pytorch geometric 
def transform_coo_to_edge_index(adj):
    adj = adj.coalesce()
    edge_index = adj.indices().detach().long()
    return edge_index

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_adj_to_scipy(adj):

    shape = adj.shape
    coords = adj.coalesce().indices()
    values = adj.coalesce().values()

    scipy_sparse = sp.coo_matrix((values.cpu().numpy(), (coords[0].cpu().numpy(), coords[1].cpu().numpy())), shape=shape)

    return scipy_sparse

# determine one edge in edge_index or not of torch geometric
def is_edge_in_edge_index(edge_index, source, target):
    mask = (edge_index[0] == source) & (edge_index[1] == target)
    return mask.any()

def construct_pseudo_assignment(cluster_ids_x):
    pseudo_assignment = torch.zeros(cluster_ids_x.shape[0], int(cluster_ids_x.max()+1))

    for i in range(cluster_ids_x.shape[0]):
        pseudo_assignment[i][int(cluster_ids_x[i])] = 1
    
    return pseudo_assignment

def pq_computation(similarity):
    q = torch.nn.functional.normalize(similarity, dim=1, p=1)
    p_temp = torch.mul(q, q)
    q_colsum = torch.sum(q, axis=0)
    p_temp = torch.div(p_temp,q_colsum)
    p = torch.nn.functional.normalize(p_temp, dim=1, p=1)
    return q, p

def coo_matrix_to_nx_graph(matrix):
    # Create an empty NetworkX graph
    graph = nx.Graph()

    # Get the number of nodes in the COO matrix
    num_nodes = matrix.shape[0]

    # Convert the COO matrix to a dense matrix
    dense_matrix = matrix.to_dense()

    # Iterate over the non-zero entries in the dense matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if dense_matrix[i][j] != 0:
                # Add an edge to the NetworkX graph
                graph.add_edge(i, j)
                graph.add_edge(j, i)

    return graph

def coo_matrix_to_nx_graph_efficient(adj_matrix):
    # 创建一个无向图对象
    graph = nx.Graph()

    # 获取 COO 矩阵的行和列索引以及权重值
    adj_matrix = adj_matrix.coalesce()
    rows = adj_matrix.indices()[0]
    cols = adj_matrix.indices()[1]

    # 添加节点和边到图中
    for i in range(len(rows)):
        graph.add_edge(int(rows[i]), int(cols[i]))
        graph.add_edge(int(cols[i]), int(rows[i]))

    return graph

def obtain_adj_from_nx(graph):
    return np.array(nx.adjacency_matrix(graph, nodelist=[i for i in range(max(graph.nodes)+1)]).todense())

def find_all_neighbors_bynx(query, Graph):
    
    nodes = Graph.nodes()

    neighbors = []
    for i in range(len(query)):
        if query[i] not in nodes:
            continue
        for j in Graph.neighbors(query[i]):
            if j not in query:
                neighbors.append(j)
    return neighbors

def MaxMinNormalization(x, Min, Max):
    
    x = np.array(x)
    x_max = np.max(x)
    x_min = np.min(x)

    x = [(item-x_min)*(Max-Min)/(x_max - x_min) + Min for item in x]

    return x


def get_silhouette_score(feats, labels, goal=1.0):
    device, dtype = feats.device, feats.dtype
    unique_labels = torch.unique(labels)
    num_samples = len(feats)
    if not (1 < len(unique_labels) < num_samples):
        raise ValueError("num unique labels must be > 1 and < num samples")

    scores = []
    score_dict = {}

    for L in unique_labels:
        if L < 0:
            continue
        curr_cluster = feats[labels == L]
        num_elements = len(curr_cluster)
        if num_elements > 1:
            intra_cluster_dists = torch.cdist(curr_cluster, curr_cluster)
            mean_intra_dists = torch.sum(intra_cluster_dists, dim=1) / (
                num_elements - 1
            )  # minus 1 to exclude self distance
            dists_to_other_clusters = []
            for otherL in unique_labels:
                if otherL != L:
                    other_cluster = feats[labels == otherL]
                    inter_cluster_dists = torch.cdist(curr_cluster, other_cluster)
                    mean_inter_dists = torch.sum(inter_cluster_dists, dim=1) / (
                        len(other_cluster)
                    )
                    dists_to_other_clusters.append(mean_inter_dists)
            dists_to_other_clusters = torch.stack(dists_to_other_clusters, dim=1)
            min_dists, _ = torch.min(dists_to_other_clusters, dim=1)
            curr_scores = (min_dists - mean_intra_dists) / (
                torch.maximum(min_dists, mean_intra_dists)
            )
        else:
            curr_scores = torch.tensor([0], device=device, dtype=dtype)

        with torch.no_grad():
            score_dict[f'silhouette/clust{L}'] = torch.mean(curr_scores).item()
            score_dict[f'clust/size_{L}'] = num_elements
        scores.append(curr_scores)

    scores = torch.cat(scores, dim=0)
    # if len(scores) != num_samples:
    #     raise ValueError(
    #         f"scores (shape {scores.shape}) should have same length as feats (shape {feats.shape})"
    #     )
    mean_score = torch.mean(scores)
    goal_diff = goal - mean_score

    score_dict['silhouette_base_score'] = mean_score.item()
    score_dict['goal_diff'] = goal_diff.item()

    return abs(goal_diff)

# returns a negative version of the VRC index
# (suitable for minimization)
def vrc_index(feats: torch.Tensor, labels: torch.Tensor):
    device, dtype = feats.device, feats.dtype
    unique_labels = torch.unique(labels)
    k = (unique_labels >= 0).sum()

    num_samples = len(feats)
    if not (1 < len(unique_labels) < num_samples):
        raise ValueError("num unique labels must be > 1 and < num samples")

    extra_disp, intra_disp = 0.0, 0.0
    mean = torch.mean(feats, dim=0)

    for k in range(k):
        cluster_k = feats[labels == k]
        mean_k = torch.mean(cluster_k, dim=0)
        extra_disp += len(cluster_k) * torch.sum((mean_k - mean) ** 2)
        intra_disp += torch.sum((cluster_k - mean_k) ** 2)

    vrc_index = extra_disp * (num_samples - k) / (intra_disp * (k - 1.0))*0.1
    return vrc_index


def get_fast_silhouette(feats: torch.Tensor, labels: torch.Tensor, goal=1.0):
    device, dtype = feats.device, feats.dtype
    unique_labels = torch.unique(labels)
    k = (unique_labels >= 0).sum()
    unique_label_idxs = torch.arange(k, device=device)
    num_samples = len(feats)
    if not (1 < len(unique_labels) < num_samples):
        raise ValueError("num unique labels must be > 1 and < num samples")

    scores = []
    score_dict = {}
    # note: we have to compute this since we need gradients
    cluster_centroids = []

    for L in unique_labels:
        if L < 0:
            continue
        curr_cluster = feats[labels == L]
        num_elements = len(curr_cluster)
        cluster_centroids.append(curr_cluster.mean(dim=0))
    cluster_centroids = torch.vstack(cluster_centroids)

    for Li in unique_label_idxs:
        L = unique_labels[Li]
        curr_cluster = feats[labels == L]
        dists = torch.cdist(curr_cluster, cluster_centroids)
        a = dists[:, Li]
        selector = torch.ones(k, dtype=torch.bool, device=device)
        selector[Li] = False
        b = torch.min(dists[:, selector], dim=1).values
        sils = (b - a) / torch.max(a, b)
        scores.append(sils)

        # with torch.no_grad():
        #     score_dict[f'silhouette/clust{L}'] = torch.mean(curr_scores).item()
        #     score_dict[f'clust/size_{L}'] = num_elements

    scores = torch.cat(scores, dim=0)
    # if len(scores) != num_samples:
    #     raise ValueError(
    #         f"scores (shape {scores.shape}) should have same length as feats (shape {feats.shape})"
    #     )
    mean_score = torch.mean(scores)
    goal_diff = (goal - mean_score)*0.1

    score_dict['silhouette_base_score'] = mean_score.item()
    score_dict['goal_diff'] = goal_diff.item()

    return abs(goal_diff)


def subgraph(adj,features):
    edge_index = transform_coo_to_edge_index(adj)
    sub_graph = Subgraph(x=features, edge_index=edge_index)
    return sub_graph.search(adj)