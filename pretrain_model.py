import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.cluster import KMeans

import utils
from early_stop import Stop_args, EarlyStopping
from layer import GraphConvolution
from deeprobust.graph.utils import to_tensor, normalize_adj_tensor
import torch.optim as optim
import numpy as np
from selfsl import *
from selfsl import NaiveGate
from metric import cluster_accuracy
from losses import re_loss_func, dis_loss


class EmptyData:

    def __init__(self):
        self.adj = None
        self.features = None
        self.adj_norm = None
        self.features_norm = None
        self.labels = None
        self.adj_label = None
        self.norm = None
        self.pos_weight = None


class PretrainModel(nn.Module):
    def __init__(self, data,encoder, modeldis, set_of_ssl, args, device, **kwargs):
        super(PretrainModel, self).__init__()
        self.args = args
        self.n_tasks = len(set_of_ssl)
        self.set_of_ssl = set_of_ssl
        self.name = "MoeSSL"
        self.device = device
        self.encoder = encoder.to(self.device)
        self.modeldis = modeldis.to(self.device)
        self.weight = torch.ones(len(set_of_ssl)).to(self.device)
        self.gate = NaiveGate(args.hid_dim[1], self.n_tasks, self.args.top_k).to(self.device)
        self.ssl_agent = []
        self.optimizer_dec = None
        self.optimizer_dis = None

        self.data = data
        self.processed_data = EmptyData()
        self.data_process()
        self.params = None
        self.setup_ssl(set_of_ssl)

    def data_process(self):
        features, adj, labels = to_tensor(self.data.features,
                                          self.data.adj, self.data.labels, device=self.device)
        adj_norm = normalize_adj_tensor(adj, sparse=True)
        self.processed_data.adj_norm = adj_norm
        self.processed_data.features = features
        self.processed_data.labels = labels

        adj_label = self.data.adj + sp.eye(self.data.adj.shape[0])
        self.processed_data.adj_label = torch.FloatTensor(adj_label.toarray()).to(self.device)
        self.processed_data.pos_weight = torch.Tensor(
            [float(self.data.adj.shape[0] * self.data.adj.shape[0] - self.data.adj.sum()) / self.data.adj.sum()]).to(
            self.device)
        norm = self.data.adj.shape[0] * self.data.adj.shape[0] / float(
            (self.data.adj.shape[0] * self.data.adj.shape[0] - self.data.adj.sum()) * 2)
        self.processed_data.norm = norm

    def setup_ssl(self, set_of_ssl):
        args = self.args
        self.params = list(self.encoder.parameters())
        for ix, ssl in enumerate(set_of_ssl):

            agent = eval(ssl)(data=self.data,
                              processed_data=self.processed_data,
                              encoder=self.encoder,
                              nhid1=self.args.hid_dim[0],
                              nhid2=self.args.hid_dim[1],
                              dropout=0.0,
                              device=self.device,
                              args=args).to(self.device)

            self.ssl_agent.append(agent)
            if agent.disc1 is not None:
                self.params = self.params + list(agent.disc1.parameters())

        self.optimizer_dis = optim.Adam(self.modeldis.parameters(), lr=0.001, betas=(0.5, 0.9))
        self.optimizer_TotalSSL = optim.Adam(self.params, lr=args.lr_pretrain, weight_decay=0.0)

    def TotalSSLpretrain(self, verbose=True):
        features = self.processed_data.features
        adj_norm = self.processed_data.adj_norm
        n_nodes, feat_dim = features.shape
        args = self.args
        arga = 1

        stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
        early_stopping = EarlyStopping(self.encoder, **stopping_args)
        loss_pretrain = []
        print(f'start pretraining')
        for i in range(self.args.pretrain_epochs):
            self.encoder.train()
            self.optimizer_TotalSSL.zero_grad()
            hidden, recovered, mu, logvar, z = self.encoder(features, adj_norm)
            if arga == 1:
                self.modeldis.train()
                for j in range(10):
                    z_real_dist = np.random.randn(self.data.adj.shape[0], args.hid_dim[1])
                    z_real_dist = torch.FloatTensor(z_real_dist)
                    z_real_dist = z_real_dist.to(self.device)
                    d_real = self.modeldis(z_real_dist)
                    d_fake = self.modeldis(mu)
                    self.optimizer_dis.zero_grad()
                    dis_loss_ = dis_loss(d_real, d_fake)
                    dis_loss_.backward(retain_graph=True)
                    self.optimizer_dis.step()
            re_loss = re_loss_func(preds=self.encoder.dc(z), labels=self.processed_data.adj_label, n_nodes=n_nodes,
                                   norm=self.processed_data.norm,
                                   pos_weight=self.processed_data.pos_weight, mu=mu, logvar=logvar, )

            ssl_loss = self.get_ssl_loss_stage_one(hidden)
            loss = args.w_ssl_stage_one * ssl_loss + re_loss

            if i % 1 == 0 and verbose:
                print(f'Epoch {i}, total loss: {loss.item()}, ssl_loss: {ssl_loss.item()}, re_loss: {re_loss.item()}')

            loss.backward()
            self.optimizer_TotalSSL.step()
            loss_pretrain.append(loss)
            if early_stopping.simple_check(loss_pretrain):
                break
        print(f'end pretraining')
        self.encoder.eval()
        # mu: (num_nodes, hidden_dim[2])
        _, _, mu, _, _ = self.encoder(features, adj_norm)
        return mu

    def FeatureFusionForward(self, z_embeddings):
        nodes_weight_ori, loss_balan = self.gate(z_embeddings, 0)
        ssl_embeddings_list = []
        for ix, ssl in enumerate(self.ssl_agent):
            ssl.train()
            ssl_embeddings = ssl.gcn2_forward(z_embeddings, self.processed_data.adj_norm)
            ssl_embeddings_list.append(ssl_embeddings)
        ssl_emb_tensor = torch.stack(ssl_embeddings_list)
        ssl_emb_tensor = ssl_emb_tensor.transpose(0, 1)
        nodes_weight = nodes_weight_ori.unsqueeze(2).expand([z_embeddings.shape[0], 5, self.args.hid_dim[1]])
        fusion_emb = torch.sum(torch.mul(ssl_emb_tensor, nodes_weight), dim=1)
        return fusion_emb, loss_balan, nodes_weight_ori

    def evaluate_pretrained(self, embeddings):
        x = embeddings.detach()
        kmeans_input = x.cpu().numpy()
        nclass = self.data.labels.max().item() + 1
        kmeans = KMeans(n_clusters=nclass, random_state=0).fit(kmeans_input)
        pred = kmeans.predict(kmeans_input)
        acc, nmi, f1 = cluster_accuracy(pred, self.data.labels, nclass)

        return acc, nmi, f1, pred, kmeans

    def bottom_features(self):
        features = self.processed_data.features_norm
        adj_norm = self.processed_data.adj_norm
        hidden, _, mu, _, _ = self.encoder(features, adj_norm)
        return hidden

    def set_weight(self, values):
        self.weight = torch.FloatTensor(values).to(self.device)

    def get_ssl_loss_stage_one(self, x):
        loss = 0
        for ix, ssl in enumerate(self.ssl_agent):
            ssl.train()
            loss = loss + ssl.make_loss_stage_one(x)
        return loss

    def get_ssl_loss_stage_two(self, x, adj):
        loss = 0
        for ix, ssl in enumerate(self.ssl_agent):
            ssl.train()
            ssl_loss = ssl.make_loss_stage_two(x, adj)
            loss = loss + ssl_loss
        return loss
class VGAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(VGAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)   # mu
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)   # logvar
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return hidden1, self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        hidden, mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return hidden, self.dc(z), mu, logvar, z


class Discriminator(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim3, hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim1, 1),
        )

    def forward(self, x):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        z = self.dis(x)
        return z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
