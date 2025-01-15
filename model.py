import torch
import math
import torch.nn as nn
from layer import TransformerBlock,VGAE
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GCNConv
from selfsl import *
from utils import is_edge_in_edge_index

class Model(nn.Module):
    def __init__(self, input_dim, args, pretrain_model=None):
        super().__init__()
        self.input_dim = input_dim
        self.args = args

        self.Linear1 = nn.Linear(input_dim, self.args.hidden_dim)
        self.encoder = TransformerBlock(hops=args.hops,
                                        input_dim=input_dim,
                                        n_layers=args.n_layers,
                                        num_heads=args.n_heads,
                                        hidden_dim=args.hidden_dim,
                                        dropout_rate=args.dropout,
                                        attention_dropout_rate=args.attention_dropout)
        if args.readout == "sum":
            self.readout = global_add_pool
        elif args.readout == "mean":
            self.readout = global_mean_pool
        elif args.readout == "max":
            self.readout = global_max_pool
        else:
            raise ValueError("Invalid pooling type.")
        
        self.marginloss = nn.MarginRankingLoss(0.5)
        # self.gate = NaiveGate(input_dim,len(set_of_ssl) , self.args.top_k)
        self.pretrain_model = pretrain_model


    def forward(self, x):
        # todo 增加moe混合层
        embeddings = torch.split(x, 1, dim=1)
        embeddings = [t.squeeze(1) for t in embeddings]
        fusion_embedding_list = []
        for embedding in embeddings:
            fusion_emb,_,_ = self.pretrain_model.FeatureFusionForward(embedding)
            fusion_embedding_list.append(fusion_emb)
        x = torch.stack(fusion_embedding_list,dim=1)
        node_tensor, neighbor_tensor = self.encoder(x) # (batch_size, 1, hidden_dim), (batch_size, hops, hidden_dim)
        neighbor_tensor = self.readout(neighbor_tensor, torch.tensor([0]).to(self.args.device)) # (batch_size, 1, hidden_dim)
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
        TotalLoss += self.args.alpha*link_loss

        return TotalLoss
