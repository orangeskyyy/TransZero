import utils
from pretrain_model import PretrainModel
from pretrain_model import VGAE,Discriminator
from deeprobust.graph.data import Dataset
import torch
import torch_geometric.transforms as T
import scipy.sparse as sp
import numpy as np

import os.path as osp
import argparse

class pyg2dpr(Dataset):
    def __init__(self, adj,features,labels):


        self.adj = utils.transform_coo_to_csr(adj)
        self.features = features.numpy()
        self.labels = labels.numpy()

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1)

def train(args,adj,features,labels):
    feat_dim = features.shape[1]
    data = pyg2dpr(adj,features,labels)
    encoder = VGAE(feat_dim, args.hid_dim[0], args.hid_dim[1], 0.0)
    modeldis = Discriminator(args.hid_dim[0], args.hid_dim[1], args.hid_dim[2])
    set_of_ssl = ['PairwiseAttrSim', 'PairwiseDistance', 'Par', 'Clu', 'DGI']
    if adj.shape[0] > 5000:
        print("use the DGISample")
        local_set_of_ssl = [ssl if ssl != 'DGI' else 'DGISample' for ssl in set_of_ssl]
    else:
        local_set_of_ssl = set_of_ssl
    pretrain_model = PretrainModel(data, feat_dim, encoder, modeldis, local_set_of_ssl, args, device=args.device).to(args.device)
    pretrain_model.train()
    embedding = pretrain_model.TotalSSLpretrain().to(args.device)
    embedding = embedding.to(args.device)
    pretrain_model.eval()
    fusion_emb, _, _ = pretrain_model.FeatureFusionForward(embedding)
    return fusion_emb


if __name__ == '__main__':
    print("node embedding pretrain")
