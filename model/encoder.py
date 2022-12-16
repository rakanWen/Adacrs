import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, List
from tianshou.data import Batch
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from adacrs.modules.gcn import GraphEncoder
from adacrs.data.processors.lastfm_graph import LastFmGraph
from adacrs.data.processors.yelp_graph import YelpGraph
from dgl import DGLHeteroGraph


class State_Encoder(nn.Module):

    def __init__(self,
                 action_size: int,
                 graph: DGLHeteroGraph,
                 device: str,
                 kg: Union[LastFmGraph, YelpGraph],
                 entity: int,
                 emb_size: int,
                 padding_id: int,
                 embeddings: torch.Tensor = None,
                 fix_emb: bool = True,
                 seq: str = 'transformer',
                 gcn: bool = True,
                 hidden_size: int = 125,
                 up_limit: int = None,
                 layers: int = 1,
                 rnn_layer: int = 1):
        super(State_Encoder, self).__init__()
        self.gcn_net = GraphEncoder(
            graph=graph,
            device=device,
            entity=entity,
            emb_size=emb_size,
            kg=kg,
            embeddings=embeddings,
            fix_emb=fix_emb,
            seq=seq,
            gcn=gcn,
            hidden_size=hidden_size - 6,  # hidden_size should be divisible by the head_number
            layers=layers,
            rnn_layer=rnn_layer)
        
        self.device = device
        self.padding_id = padding_id
        
        self.gcn_net.embedding = nn.Embedding(entity, emb_size, padding_idx=padding_id)
        if embeddings is not None:
            print("pre-trained embeddings")
            self.gcn_net.embedding.from_pretrained(embeddings, freeze=fix_emb)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        # this out net is ask property or item
        self.out = nn.Linear(hidden_size, 2)
        self.out.weight.data.normal_(0, 0.1)  # initialization
        # V(s)
        self.fc2_value = nn.Linear(hidden_size, hidden_size)
        self.out_value = nn.Linear(hidden_size, 1)
        # Q(s,a)
        self.fc2_advantage = nn.Linear(hidden_size + action_size, hidden_size)
        self.out_advantage = nn.Linear(hidden_size, 1)
        self.up_limit = up_limit

    def forward(self,
                graph_state: List[Union[Batch, Dict]],
                scpr_emb: List[torch.Tensor],
                cand_item: List[np.ndarray] = None,
                cand_feat: List[np.ndarray] = None,
                len_feature: int = None,
                action_batch: torch.LongTensor = None,
                choose_action: bool = True,
                is_list: bool = True,
                turn: int = 0):
        """
        :param x: encode history [N*L*D]; y: action embedding [N*K*D]
        :return: v: action score [N*K]
        """
        state_emb_batch = self.gcn_net(graph_state)
        if is_list is True:
            scpr_emb_batch = torch.stack(scpr_emb).view(
                state_emb_batch.shape[0], state_emb_batch.shape[1], -1).cuda()
        else:  # select action
            scpr_emb_batch = torch.tensor(scpr_emb).view(
                state_emb_batch.shape[0], state_emb_batch.shape[1], -1).cuda()
        x = torch.cat((scpr_emb_batch, state_emb_batch), -1)
        if cand_item is not None:
            cand_batch = torch.cat(
                (self.padding(cand_feat), self.padding(cand_item)), dim=-1)
            len_feature = self.padding(cand_feat).shape[-1]
            y = self.gcn_net.embedding(cand_batch)
        else:
            y = self.gcn_net.embedding(action_batch)

        # print(x.shape, y.shape)

        x = self.fc1(x)
        x = F.relu(x)
        out = self.out(x).squeeze(1)
        logit_action_value_ = torch.softmax(out, dim=-1).squeeze(1)
        cat = RelaxedOneHotCategorical(0.3, logits=logit_action_value_)
        # re-parameter
        action_value_ = cat.rsample().float()
        action_value = torch.zeros_like(action_value_, device=self.device)
        action_value[:, 0] = torch.where(
            action_value_[:, 0] > action_value_[:, 1], 1, 0)
        action_value[:, 1] = torch.where(action_value[:, 0] == 1, 0, 1)
        # V
        value = self.out_value(F.relu(self.fc2_value(x))).squeeze(
            dim=2)  # [N*1*1]
        # Q(s,a)
        if choose_action:
            x = x.repeat(1, y.size(1), 1)
        state_cat_action = torch.cat((x, y), dim=2)
        # A
        advantage = self.out_advantage(
            F.relu(self.fc2_advantage(state_cat_action))).squeeze(
                dim=2)  # [N*K]

        if choose_action:
            qsa = advantage + value - advantage.mean(dim=1, keepdim=True)

            qsa[:, :len_feature] *= action_value_[:, 0].unsqueeze(
                -1)  # +=(1-action_value[:,0].unsqueeze(-1))*(-1e10)
            # item
            qsa[:, len_feature:] *= action_value_[:, 1].unsqueeze(
                -1)  # +=(1-action_value[:,1].unsqueeze(-1))*(-1e10)
        else:
            qsa = advantage + value
            v = action_value.gather(
                1, torch.where(action_batch < self.up_limit, 1,
                               0))  # get state from action through gather
            qsa = qsa * v
        # print(qsa.shape)
        return action_value_, qsa, out

    def gcn_emb(self, state):
        return self.gcn_net.embedding(state)

    def gcn_(self, state):
        return self.gcn_net(state)

    def padding(self, cand: List[np.ndarray]):
        pad_size = max([len(c) for c in cand])
        padded_cand = []
        for c in cand:
            cur_size = len(c)
            new_c = np.ones((pad_size)) * self.padding_id
            new_c[:cur_size] = c
            padded_cand.append(new_c)
        return torch.LongTensor(padded_cand).to(self.device)
