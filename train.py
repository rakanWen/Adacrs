import math
import numpy as np
import os
from tqdm import tqdm
import argparse
from itertools import count
import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional
from torch.utils.tensorboard import SummaryWriter
import sys
# sys.path.append('/home/wenxiaofei.wxf/crshust')
from adacrs.utils.utils_load_save import LAST_FM, LAST_FM_STAR, YELP, YELP_STAR, TMP_DIR
from adacrs.utils.utils_load_save import set_random_seed, enablePrint, load_kg, load_dataset
# TODO select env
from adacrs.envs.binary_question import BinaryRecommendEnv
from adacrs.envs.enumerated_question import EnumeratedRecommendEnv
from adacrs.utils.evaluate import dqn_evaluate
from adacrs.utils.construct_graph import get_graph
from adacrs.data.buffer.myPriorReplayBuffer import MyPriorReplayBuffer
from tianshou.data import (Batch, to_numpy)
from adacrs.policy.agent import Agent
from adacrs.model.encoder import State_Encoder
import warnings
import logging

warnings.filterwarnings("ignore")
EnvDict = {
    LAST_FM: BinaryRecommendEnv,
    LAST_FM_STAR: BinaryRecommendEnv,
    YELP: EnumeratedRecommendEnv,
    YELP_STAR: BinaryRecommendEnv
}
FeatureDict = {
    LAST_FM: 'feature',
    LAST_FM_STAR: 'feature',
    YELP: 'large_feature',
    YELP_STAR: 'feature'
}

logger = logging.getLogger(__file__)


def train(args, kg, dataset, filename):
    env = EnvDict[args.data_name](kg,
                                  dataset,
                                  args.data_name,
                                  args.embed,
                                  seed=args.seed,
                                  max_turn=args.max_turn,
                                  cand_num=args.cand_num,
                                  cand_item_num=args.cand_item_num,
                                  attr_num=args.attr_num,
                                  mode=args.mode,
                                  ask_num=args.ask_num,
                                  entropy_way=args.entropy_method,
                                  fm_epoch=args.fm_epoch)
    set_random_seed(args.seed)
    G = get_graph(env.user_length, env.item_length, env.feature_length,
                  args.data_name)
    memory = MyPriorReplayBuffer(size=args.memory_size,
                                 alpha=0.6,
                                 beta=0.4,
                                 weight_norm=True)
    embed = torch.FloatTensor(
        np.concatenate((env.ui_embeds, env.feature_emb,
                        np.zeros((1, env.ui_embeds.shape[1]))),
                       axis=0))
    logger.info("-----import DGL successfully!!-----")
    up_limit = len(env.ui_embeds)
    model = State_Encoder(action_size=embed.size(1),
                          graph=G,
                          device=args.device,
                          kg=kg,
                          entity=env.user_length + env.item_length + env.feature_length + 1,
                          emb_size=embed.size(1),
                          padding_id=env.user_length + env.item_length + env.feature_length,
                          fix_emb=args.fix_emb,
                          seq=args.seq,
                          gcn=args.gcn,
                          hidden_size=args.hidden,
                          up_limit=up_limit).to(args.device)
    agent = Agent(model=model,
                  optim=optim.Adam(model.parameters(),
                                   lr=args.learning_rate,
                                   weight_decay=args.l2_norm),
                  discount_factor=args.gamma,
                  device=args.device,
                  memory=memory,
                  padding_id=env.user_length + env.item_length + env.feature_length,
                  up_limit=up_limit)
    if args.load_rl_epoch != 0:
        logger.info('Staring loading rl model in epoch {}'.format(
            args.load_rl_epoch))
        agent.load_model(data_name=args.data_name,
                         filename=filename,
                         epoch_user=args.load_rl_epoch)

    test_performance = []
    if args.eval_num == 1:
        SR15_mean = dqn_evaluate(args, kg, dataset, agent, filename, 0)
        test_performance.append(SR15_mean)

    if not os.path.isdir(TMP_DIR[args.data_name] + '/log/'):
        os.makedirs(TMP_DIR[args.data_name] + '/log/')
    writer = SummaryWriter(log_dir=TMP_DIR[args.data_name] + '/log/')
    count_1 = 0
    count_2 = 0

    # SR15_mean = dqn_evaluate(args, kg, dataset, agent, filename, 0)
    for train_step in range(1, args.max_steps + 1):
        SR5, SR10, SR15, AvgT, Rank, total_reward = 0., 0., 0., 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)
        for i_episode in tqdm(range(args.sample_times), desc='sampling'):
            logger.info(
                '\n================new tuple:{}===================='.format(
                    i_episode))
            if not args.fix_emb:  # Reset environment and record the starting state
                scpr, state, cand_feature, cand_item, ac_label, action_space = env.reset(
                    agent.model.gcn_net.embedding.weight.data.cpu().detach(
                    ).numpy())
            else:
                scpr, state, cand_feature, cand_item, ac_label, action_space = env.reset(
                )
            epi_reward = 0
            is_last_turn = False
            for t in count():  # user  dialog # 0-14
                if t == 14:
                    is_last_turn = True
                action, sorted_actions, is_sample = agent.select_action(
                    scpr,
                    state,
                    cand_feature,
                    cand_item,
                    action_space,
                    is_last_turn=is_last_turn,
                    turn=t)
                if not args.fix_emb:
                    next_scpr, next_state, next_cand_feature, next_cand_item,\
                        nac_label, action_space, reward, done = env.step(
                            action.item(), sorted_actions,
                            agent.model.gcn_net.embedding.weight.data.cpu().detach().numpy(),
                            is_sample=is_sample)
                else:
                    next_scpr, next_state, next_cand_feature, next_cand_item,\
                        nac_label, action_space, reward, done = env.step(
                            action.item(), sorted_actions, is_sample=is_sample)
                epi_reward += reward
                reward = torch.tensor([reward],
                                      device=args.device,
                                      dtype=torch.float)
                if done:
                    next_state = None
                if next_state is not None:
                    info_ = {
                        "scpr": torch.tensor(scpr),
                        "next_scpr": torch.tensor(next_scpr),
                        "ncand_feat": next_cand_feature,
                        "ncand_item": next_cand_item,
                        "ac_label": torch.tensor([int(ac_label)]),
                        "nac_label": torch.tensor([int(nac_label)])
                    }
                    batch = Batch(obs=state,
                                  act=to_numpy(action),
                                  rew=to_numpy(reward),
                                  done=done,
                                  obs_next=next_state,
                                  info=info_)
                    agent.memory.add(batch)
                    logger.info("finish add method")
                    state = next_state
                    cand_feature = next_cand_feature
                    cand_item = next_cand_item
                    scpr = next_scpr
                    ac_label = nac_label
                    loss_option = agent.update(args.batch_size, memory, 0)
                    loss_choice = agent.update(args.batch_size, memory, 1)
                    if loss_option is not None:
                        writer.add_scalar('loss1', loss_option.item(), count_1)
                        loss += loss_option
                        count_1 += 1
                    if loss_choice is not None:
                        writer.add_scalar('loss2', loss_choice.item(), count_2)
                        loss += loss_choice
                        count_2 += 1
                if done:
                    if reward.item() == 1:  # recommend successfully
                        if t < 5:
                            SR5 += 1
                            SR10 += 1
                            SR15 += 1
                        elif t < 10:
                            SR10 += 1
                            SR15 += 1
                        else:
                            SR15 += 1
                        Rank += (
                            1 / math.log(t + 3, 2) +
                            (1 / math.log(t + 2, 2) - 1 / math.log(t + 3, 2)) /
                            math.log(done + 1, 2))
                    else:
                        Rank += 0
                    AvgT += t + 1
                    total_reward += epi_reward
                    break
        enablePrint()  # Enable logger.info function
        logger.info('loss : {} in epoch_uesr {}'.format(
            loss.item() / args.sample_times, args.sample_times))
        logger.info('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, rewards:{} '
                    'Total epoch_uesr:{}'.format(
                        SR5 / args.sample_times, SR10 / args.sample_times,
                        SR15 / args.sample_times, AvgT / args.sample_times,
                        Rank / args.sample_times,
                        total_reward / args.sample_times, args.sample_times))

        if train_step % args.eval_num == 0:
            SR15_mean = dqn_evaluate(args, kg, dataset, agent, filename,
                                     train_step)
            test_performance.append(SR15_mean)
        if train_step % args.save_num == 0:
            agent.save_model(data_name=args.data_name,
                             filename=filename,
                             epoch_user=train_step)
    logger.info(test_performance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        '-seed',
                        type=int,
                        default=5,
                        help='random seed.')
    parser.add_argument('--gpu', type=str, default='1', help='gpu device.')
    parser.add_argument('--fm_epoch',
                        type=int,
                        default=0,
                        help='the epoch of FM embedding')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        #128
                        help='batch size.')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.99,
                        help='reward discount factor.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-4,
                        help='learning rate.')
    parser.add_argument('--l2_norm',
                        type=float,
                        default=1e-6,
                        help='l2 regularization.')
    parser.add_argument('--hidden',
                        type=int,
                        default=106,
                        help='number of samples')
    parser.add_argument('--memory_size',
                        type=int,
                        default=5000,
                        #50000
                        help='size of memory ')

    parser.add_argument('--data_name',
                        type=str,
                        default=YELP_STAR,
                        choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR],
                        help='One of {LAST_FM, LAST_FM_STAR, \
                            YELP, YELP_STAR}.')
    parser.add_argument('--entropy_method',
                        type=str,
                        default='weight_entropy',
                        help='entropy_method is one of {entropy, weight \
                            entropy}')
    parser.add_argument('--max_turn',
                        type=int,
                        default=5,
                        help='max conversation turn')
    parser.add_argument('--attr_num',
                        type=int,
                        help='the number of attributes')
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        help='the mode in [train, test]')
    parser.add_argument('--ask_num',
                        type=int,
                        default=1,
                        help='the number of features asked in a turn')
    parser.add_argument('--load_rl_epoch',
                        type=int,
                        default=0,
                        help='the epoch of loading RL model')

    parser.add_argument('--sample_times',
                        type=int,
                        default=50,
                        #100
                        help='the epoch of sampling')
    parser.add_argument('--max_steps',
                        type=int,
                        default=40,
                        help='max training steps')
    parser.add_argument('--eval_num',
                        type=int,
                        default=4,
                        help='the number of steps to \
                            evaluate RL model and metric')
    parser.add_argument('--save_num',
                        type=int,
                        default=4,
                        help='the number of steps to save RL model and metric')
    parser.add_argument('--observe_num',
                        type=int,
                        default=100,
                        #500
                        help='the number of steps to logger.info metric')
    parser.add_argument('--cand_num',
                        type=int,
                        default=1,
                        help='candidate sampling number')
    parser.add_argument('--cand_item_num',
                        type=int,
                        default=1,
                        help='candidate item sampling number') # cand_item_num + max_turn + 100 = hidden_size
    parser.add_argument('--fix_emb',
                        action='store_false',
                        help='fix embedding or not')
    parser.add_argument('--embed',
                        type=str,
                        default='transe',
                        help='pretrained embeddings')
    parser.add_argument('--seq',
                        type=str,
                        default='transformer',
                        choices=['rnn', 'transformer', 'mean'],
                        help='sequential learning method')
    parser.add_argument('--gcn', action='store_false', help='use GCN or not')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    logger.info('gpu:{}'.format(args.device))
    logger.info('batch size:{}'.format(args.batch_size))
    logger.info('data_set:{}'.format(args.data_name))
    kg = load_kg(args.data_name)
    # reset attr_num
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    logger.info('dataset:{}, feature_length:{}'.format(args.data_name,
                                                       feature_length))
    args.attr_num = feature_length  # set attr_num  = feature_length
    logger.info(f'args.attr_num: {args.attr_num}')
    logger.info(f'args.entropy_method: {args.entropy_method}')
    dataset = load_dataset(args.data_name)
    filename = 'train-data-{}-RL-cand_num-{}-cand_item_num-{}-embed-{}-seq-{}-gcn-{}'.format(
        args.data_name, args.cand_num, args.cand_item_num, args.embed,
        args.seq, args.gcn)
    train(args, kg, dataset, filename)


if __name__ == '__main__':
    main()
