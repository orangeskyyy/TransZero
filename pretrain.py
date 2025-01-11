from pretrain_model import PretrainModel
from pretrain_model import VGAE,Discriminator
from deeprobust.graph.data import Dataset
import torch
import torch_geometric.transforms as T
import scipy.sparse as sp
import numpy as np

import os.path as osp
import argparse

def train(args,adj,feature,labels):
    pretrain_parser = argparse.ArgumentParser()
    pretrain_parser.add_argument('--device', type=str, default='cuda', help='use cpu or gpu.')
    pretrain_parser.add_argument('--gpu', type=int, default=0, help='GPU id.')
    pretrain_parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    pretrain_parser.add_argument('--dataset', type=str, default='photo', help='Type of dataset.')
    pretrain_parser.add_argument('--encoder', type=str, default="ARVGA", help='the model name')
    pretrain_parser.add_argument('--hid_dim', type=int, nargs='+', default=[256, 128, 512])
    pretrain_parser.add_argument('--lr_pretrain', type=float, default=0.001, help='Initial learning rate.')
    pretrain_parser.add_argument('--pretrain_epochs', type=int, default=400, help='Number of pretrained totalssl epochs.')
    pretrain_parser.add_argument('--w_ssl_stage_one', type=float, default=0.25, help='the ssl_loss weight of pretrain stage one')
    pretrain_parser.add_argument('--use_ckpt', type=int, default=1, help='whether to use checkpoint, 0/1.')
    pretrain_parser.add_argument('--save_ckpt', type=int, default=1, help='whether to save checkpoint, 0/1.')
    pretrain_parser.add_argument('--top_k', type=int, default=5, help='The number of experts to choose.')
    pretrain_parser.add_argument('--st_per_p', type=float, default=0.5, help='The threshold of the pseudo positive labels.')
    pretrain_parser.add_argument('--lr_train_fusion', type=float, default=0.001, help='train pseudo labels learning rate.')
    pretrain_parser.add_argument('--labels_epochs', type=int, default=250, help='Number of epochs to train.')
    args = pretrain_parser.parse_args()
    feat_dim = feature.shape[1]
    encoder = VGAE(feat_dim, args.hid_dim[0], args.hid_dim[1], 0.0)
    modeldis = Discriminator(args.hid_dim[0], args.hid_dim[1], args.hid_dim[2])
    set_of_ssl = ['PairwiseAttrSim', 'PairwiseDistance', 'Par', 'Clu', 'DGI']
    if adj.shape[0] > 5000:
        print("use the DGISample")
        local_set_of_ssl = [ssl if ssl != 'DGI' else 'DGISample' for ssl in set_of_ssl]
    else:
        local_set_of_ssl = set_of_ssl
    pretrain_model = PretrainModel(adj, feature, labels, encoder, modeldis, local_set_of_ssl, args, device=args.device).to(args.device)
    pretrain_model.train()
    hidden,fusion_emb = pretrain_model.TotalSSLpretrain()
    return hidden,fusion_emb

def get_dataset(dataset, name, normalize_features=False, transform=None, if_dpr=True):
    if dataset in {"photo", "cs"}:
        file_path = "dataset/" + dataset + "_dgl.pt"
    else:
        file_path = "dataset/" + dataset + "_pyg.pt"
    # file_path = "dataset/"+dataset+".pt"
    data_list = torch.load(file_path)

    adj = data_list[0]

    features = data_list[1]


    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    if not if_dpr:
        return dataset

    return Pyg2Dpr(dataset)

class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, multi_splits=False, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass

        dataset_name = pyg_data.name
        pyg_data = pyg_data[0]
        n = pyg_data.num_nodes

        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
            (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1)

if __name__ == '__main__':
    print("node embedding pretrain")
