import math
import torch
import dgl.nn as dglnn
from typing import Dict, Union, List
from tianshou.data import Batch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn
from dgl import DGLHeteroGraph
from adacrs.data.processors.lastfm_graph import LastFmGraph
from adacrs.data.processors.yelp_graph import YelpGraph


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input,
                           self.weight)  # input:[xxx,64] self.weight:[64,100]
        output = torch.sparse.mm(adj,
                                 support)  # support:[xxx,100] adj [xxx,xxx]
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphEncoder(Module):

    def __init__(self,
                 graph: DGLHeteroGraph,
                 device: str,
                 entity: int,
                 emb_size: int,
                 kg: Union[LastFmGraph, YelpGraph],
                 embeddings=None,
                 fix_emb: bool = True,
                 seq: str = 'rnn',
                 gcn: bool = True,
                 hidden_size: int = 100,
                 layers: int = 1,
                 rnn_layer: int = 1):
        super(GraphEncoder, self).__init__()

        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        # self.eps = 0.0
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        rel_names = ['interact', 'friends', 'like', 'belong_to']
        self.G = graph.to(device)
        self.conv1 = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(emb_size, hidden_size)
             for rel in rel_names},
            aggregate='mean')

        self.embedding = nn.Embedding(entity, emb_size, padding_idx=entity - 1)
        if embeddings is not None:
            print("pre-trained embeddings")
            self.embedding.from_pretrained(embeddings, freeze=fix_emb)
        
        self.layers = layers
        self.user_num = len(kg.G['user'])
        self.item_num = len(kg.G['item'])
        self.PADDING_ID = entity - 1
        self.device = device
        self.seq = seq
        self.gcn = gcn
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc_neg = nn.Linear(hidden_size, hidden_size)
        if self.seq == 'rnn':
            self.rnn = nn.GRU(hidden_size,
                              hidden_size,
                              rnn_layer,
                              batch_first=True)
        elif self.seq == 'transformer':
            self.transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size,
                                                         nhead=4,
                                                         dim_feedforward=400),
                num_layers=rnn_layer)

        if self.gcn:
            indim, outdim = emb_size, hidden_size  # 64, 100
            self.gnns = nn.ModuleList()
            for _ in range(layers):
                self.gnns.append(GraphConvolution(indim, outdim))
                indim = outdim
        else:
            self.fc2 = nn.Linear(emb_size, hidden_size)

    def forward(self, b_state: List[Union[Batch, Dict]]):
        """
        :param b_state [N]
        :return: [N x L x d]
        """
        # from www
        h0 = {
            'user':
            self.embedding(
                torch.arange(0, self.user_num).long().to(self.device)),
            'item':
            self.embedding(
                torch.arange(self.user_num, self.user_num +
                             self.item_num).long().to(self.device)),
            'attribute':
            self.embedding(
                torch.arange(self.user_num + self.item_num,
                             self.PADDING_ID + 1).long().to(self.device))
        }
        h1 = self.conv1(self.G, h0)
        h1 = {k: self.relu((v)) for k, v in h1.items()}

        gnn_embedding = torch.cat((h1['user'], h1['item']), dim=0)
        gnn_embedding = torch.cat((gnn_embedding, h1['attribute']), dim=0)
        # from www

        batch_output = []
        for s in b_state:
            # neighbors, adj = self.get_state_graph(s)
            neighbors, adj = s['neighbors'].to(self.device), s['adj'].to(
                self.device)
            input_state = self.embedding(
                neighbors.squeeze(dim=0))  # neighbors.size():[1,xxx]
            if self.gcn:
                for gnn in self.gnns:
                    output_state = gnn(input_state, adj)
                    input_state = output_state
                batch_output.append(output_state)
            else:
                output_state = F.relu(self.fc2(input_state))
                batch_output.append(output_state)
        # from www
        seq_embeddings = []
        rej_feature_embeddings = []
        rej_item_embeddings = []
        user_em = []
        for s, o in zip(b_state, batch_output):
            # seq_embeddings.append(o[:len(s['cur_node']),:][None,:])
            seq_embeddings.append((1 - self.eps) *
                                  o[:len(s['cur_node']), :][None, :] +
                                  self.eps * gnn_embedding[s['cur_node']])
            if len(s['rej_feature']) > 0:
                rej_feature_embeddings.append(
                    torch.mean(gnn_embedding[s['rej_feature']],
                               dim=0).view(1, -1))
            else:
                rej_feature_embeddings.append(
                    torch.zeros([1, self.hidden_size]).to(self.device))
            if len(s['rej_item']) > 0:
                rej_item_embeddings.append(
                    torch.mean(gnn_embedding[s['rej_item']],
                               dim=0).view(1, -1))
            else:
                rej_item_embeddings.append(
                    torch.zeros([1, self.hidden_size]).to(self.device))
            user_em.append(gnn_embedding[s['user']])  # from www

        if len(batch_output) > 1:
            seq_embeddings = self.padding_seq(seq_embeddings)
        # if len(seq_embeddings) > 0: #???
        seq_embeddings = torch.cat(seq_embeddings, dim=0)  # [N x L x d]
        # if len(user_em) > 0:
        user_em = torch.cat(user_em, dim=0).view(-1, 1, self.hidden_size)
        rej_embedding = None

        # interest_emb=torch.cat((user_em,seq_embeddings),1)
        interest_emb = seq_embeddings
        if len(rej_feature_embeddings) > 0 and len(rej_item_embeddings) > 0:
            rej_feature_embed = torch.cat(rej_feature_embeddings,
                                          dim=0).unsqueeze(1)
            rej_item_embed = torch.cat(rej_item_embeddings, dim=0).unsqueeze(1)
            rej_embedding = rej_item_embed + rej_feature_embed
        elif len(rej_feature_embeddings) > 0:
            rej_feature_embed = torch.cat(rej_feature_embeddings,
                                          dim=0).unsqueeze(1)
            rej_embedding = rej_feature_embed
        elif len(rej_item_embeddings) > 0:
            rej_item_embed = torch.cat(rej_item_embeddings, dim=0).unsqueeze(1)
            rej_embedding = rej_item_embed
        if rej_embedding is not None:
            mm = F.relu(self.fc_neg(rej_embedding))
            interest_emb = torch.cat((interest_emb, mm), 1)

        if self.seq == 'rnn':
            _, h = self.rnn(seq_embeddings)
            seq_embeddings = h.permute(1, 0, 2)  # [N*1*D]
        elif self.seq == 'transformer':
            seq_embeddings = torch.mean(self.transformer(seq_embeddings),
                                        dim=1,
                                        keepdim=True)
        elif self.seq == 'mean':
            seq_embeddings = torch.mean(seq_embeddings, dim=1, keepdim=True)

        seq_embeddings = F.relu(self.fc1(seq_embeddings))

        return seq_embeddings

    def padding_seq(self, seq: List[torch.Tensor]):
        padding_size = max([len(x[0]) for x in seq])
        padded_seq = []
        for s in seq:
            cur_size = len(s[0])
            emb_size = len(s[0][0])
            new_s = torch.zeros((padding_size, emb_size)).to(self.device)
            new_s[:cur_size, :] = s[0]
            padded_seq.append(new_s[None, :])
        return padded_seq
