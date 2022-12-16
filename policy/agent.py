import math
from typing import Dict, List, Optional, Union
import random
import numpy as np
import torch
import torch.nn as nn
from adacrs.data.buffer.myPriorReplayBuffer import MyPriorReplayBuffer
from adacrs.utils.utils_load_save import load_rl_agent, save_rl_agent
from tianshou.policy import DQNPolicy
from tianshou.data import Batch
from tianshou.data import PrioritizedReplayBuffer

class Agent(DQNPolicy):

    def __init__(self,
                 model: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                 discount_factor: float,
                 device: str,
                 memory: PrioritizedReplayBuffer,
                 padding_id: int,
                 up_limit: int = None,
                 estimation_step: int = 1,
                 target_update_freq: int = 1,
                 reward_normalization: bool = False,
                 is_double: bool = False,
                 clip_loss_grad: bool = False,
                 eps_start: float = 0.9,
                 eps_end: float = 0.1,
                 eps_decay: float = 0.0001,
                 tau: float = 0.01):
        super(Agent, self).__init__(model=model,
                                    optim=optim,
                                    discount_factor=discount_factor,
                                    estimation_step=estimation_step,
                                    target_update_freq=target_update_freq,
                                    reward_normalization=reward_normalization,
                                    is_double=is_double,
                                    clip_loss_grad=clip_loss_grad)
        self.steps_done = 0
        self.device = device
        # learn
        self.memory = memory
        self.padding_id = padding_id
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.up_limit = up_limit

    def select_action(self,
                      scpr: List[int],
                      state: Dict[str, Union[torch.Tensor, int, List[int]]],
                      cand_feature: List[int],
                      cand_item: List[int],
                      action_space: List[int],
                      is_test: bool = False,
                      is_last_turn: bool = False,
                      turn: int = 0):
        cand = cand_feature + cand_item
        len_feature = len(cand_feature)
        cand = torch.LongTensor([cand]).to(self.device)
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)  # forever 0.1
        self.steps_done += 1
        if is_test or sample > eps_threshold:  # predict with experience
            if is_test and (len(action_space[1]) <= 10 or is_last_turn):
                return torch.tensor(action_space[1][0],
                                    device=self.device,
                                    dtype=torch.long), action_space[1], False
            self.model.eval()
            with torch.no_grad():
                _, actions_value, out = self.model(graph_state=[state],
                                                   scpr_emb=scpr,
                                                   action_batch=cand,
                                                   len_feature=len_feature,
                                                   cand_item=None,
                                                   cand_feat=None,
                                                   is_list=False,
                                                   turn=turn)
                print(sorted(list(zip(cand[0].tolist(), actions_value[0].tolist())), 
                           key=lambda x: x[1], reverse=True))
                action = cand[0][actions_value.argmax().item()]

                sorted_actions = cand[0][actions_value.sort(1,
                                                            True)[1].tolist()]
                return action, sorted_actions.tolist(), False
        else:  # explore
            shuffled_cand = action_space[0] + action_space[1]  # item + feature
            random.shuffle(shuffled_cand)
            return torch.tensor(shuffled_cand[0],
                                device=self.device,
                                dtype=torch.long), shuffled_cand, True
            # exploration_noise

    def update_target_model(self):
        # soft assign
        for target_param, param in zip(self.model_old.parameters(),
                                       self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data *
                                    (1.0 - self.tau))

    def learn(self, batch_size: int, batch: Batch, epoch: int,
              idxs: np.ndarray):
        """
        : batch only for _reserved_keys:("obs", "act", "rew", "done", "obs_next", "info", "policy")
        """
        if len(self.memory) < batch_size:
            return
        self.update_target_model()
        self.optim.zero_grad()
        is_weights = self.memory.get_weight(idxs)

        #  --- first policy select choose property or item ---
        # ---- second policy select choose what property or choose what item
        try:
            action_batch = torch.LongTensor(
                np.array(batch.act).astype(int).reshape(-1, 1)).to(
                    self.device)  # [N*1]
        except ValueError:
            print(batch.act)
            raise ValueError

        reward_batch = torch.FloatTensor(
            np.array(batch.rew).astype(float).reshape(-1, 1)).to(self.device)
        non_final_mask = torch.tensor(tuple(
            map(lambda s: s is not None, batch.obs_next)),
                                      device=self.device,
                                      dtype=torch.uint8)
        action_value, q_eval, out = self.model(
            graph_state=batch.obs,
            scpr_emb=batch.info.scpr.tolist(),
            action_batch=action_batch,
            choose_action=False)
        # reward and action batch data
        n_states = []
        n_cands_feature = []
        n_cands_item = []
        nac_label = []
        n_scpr_state = []
        for ns, ncf, nci, nl, nscp in zip(batch.obs_next,
                                          batch.info.ncand_feat,
                                          batch.info.ncand_item,
                                          batch.info.nac_label,
                                          batch.info.next_scpr):
            if ns is not None:
                n_states.append(ns)
                n_cands_feature.append(ncf)
                n_cands_item.append(nci)
                nac_label.append(nl)
                n_scpr_state.append(nscp)
        nac_label = torch.tensor(nac_label, device=self.device)

        next_cand_feature_batch = self.padding(n_cands_feature)
        next_cand_item_batch = self.padding(n_cands_item)
        next_cand_batch = torch.cat(
            (next_cand_feature_batch, next_cand_item_batch), dim=-1)
        # next step represetation

        # Double DQN

        best_actions = torch.gather(
            input=next_cand_batch,
            dim=1,
            index=self.model(graph_state=n_states,
                             scpr_emb=n_scpr_state,
                             cand_item=n_cands_item,
                             cand_feat=n_cands_feature)[1].argmax(dim=1).view(
                                 len(n_states), 1).to(self.device))

        q_target = torch.zeros((batch_size, 1), device=self.device)
        action_value_n, qqq, out_n = self.model_old(n_states,
                                                    n_scpr_state,
                                                    action_batch=best_actions,
                                                    choose_action=False)

        loss_func = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()

        state_action_value = action_value.gather(
            1, torch.where(action_batch < self.up_limit, 1, 0))

        q_target[non_final_mask] = qqq.detach()
        q_target = reward_batch + self._gamma * q_target  # process_fn(compute_nstep_return) for tianshou
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = action_value_n.max(1)[0].detach()
        state_value_target = reward_batch + self._gamma * next_state_values

        errors = (q_eval - q_target).detach().cpu().squeeze().tolist()
        batch.weight = errors  # prio replaybuffer weight

        ac_label = torch.tensor(batch.info.ac_label.tolist()).to(self.device)
        pre_score = torch.cat([action_value, action_value_n], dim=0)
        ground = torch.cat([ac_label, nac_label], dim=-1)

        a_loss = criterion(pre_score, ground)
        if epoch == 0:
            loss = a_loss + (
                torch.FloatTensor(is_weights).to(self.device) *
                loss_func(state_action_value, state_value_target)).mean()
        else:
            print('loss:', (torch.FloatTensor(is_weights).to(self.device) *
                            loss_func(q_eval, q_target)).mean())
            loss = (torch.FloatTensor(is_weights).to(self.device) *
                    loss_func(q_eval, q_target)).mean()
        loss.backward()
        self.optim.step()

        return loss.data

    def update(self, batch_size: int, buffer: Optional[MyPriorReplayBuffer],
               epoch: int):  # batch keyword
        if buffer is None:
            return {}
        batch_, indices = buffer.sample(batch_size=batch_size)  # sample batch
        self.updating = True
        result = self.learn(batch_size=batch_size,
                            batch=batch_,
                            epoch=epoch,
                            idxs=indices)
        self.post_process_fn(batch_, buffer, indices)
        self.updating = False
        return result

    def save_model(self, data_name: str, filename: str, epoch_user: int):
        save_rl_agent(dataset=data_name,
                      model={
                          'policy': self.model.state_dict(),
                          'gcn': self.model.gcn_net.state_dict()
                      },
                      filename=filename,
                      epoch_user=epoch_user)

    def load_model(self, data_name: str, filename: str, epoch_user: int):
        model_dict = load_rl_agent(dataset=data_name,
                                   filename=filename,
                                   epoch_user=epoch_user)
        self.model.load_state_dict(model_dict['policy'])
        self.model.gcn_net.load_state_dict(model_dict['gcn'])

    def padding(self, cand: List[np.ndarray]):
        pad_size = max([len(c) for c in cand])
        padded_cand = []
        for c in cand:
            cur_size = len(c)
            new_c = np.ones((pad_size)) * self.padding_id
            new_c[:cur_size] = c
            padded_cand.append(new_c)
        return torch.LongTensor(padded_cand).to(self.device)
