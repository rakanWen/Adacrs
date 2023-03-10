import time
from itertools import count
import torch
import math
import numpy as np
from typing import Union
from adacrs.utils.utils_load_save import set_random_seed, enablePrint, save_rl_mtric
from adacrs.utils.utils_load_save import LAST_FM, LAST_FM_STAR, YELP, YELP_STAR, INS, TMP_DIR
from envs.binary_question import BinaryRecommendEnv
from envs.enumerated_question import EnumeratedRecommendEnv
from tqdm import tqdm
from adacrs.policy.agent import Agent
from adacrs.data.processors.lastfm_graph import LastFmGraph
from adacrs.data.processors.yelp_graph import YelpGraph
# from adacrs.data.processors.base import Graph
# from adacrs.data.processors.base import Processor
EnvDict = {
    LAST_FM: BinaryRecommendEnv,
    LAST_FM_STAR: BinaryRecommendEnv,
    YELP: EnumeratedRecommendEnv,
    YELP_STAR: BinaryRecommendEnv,
    INS: BinaryRecommendEnv
}


def dqn_evaluate(args, kg: Union[LastFmGraph, YelpGraph], dataset, agent: Agent, filename: str, i_episode: int):
    test_env = EnvDict[args.data_name](kg,
                                       dataset,
                                       args.data_name,
                                       args.embed,
                                       seed=args.seed,
                                       max_turn=args.max_turn,
                                       cand_num=args.cand_num,
                                       cand_item_num=args.cand_item_num,
                                       attr_num=args.attr_num,
                                       mode='test',
                                       ask_num=args.ask_num,
                                       entropy_way=args.entropy_method,
                                       fm_epoch=args.fm_epoch)
    set_random_seed(args.seed)
    tt = time.time()
    start = tt
    SR5, SR10, SR15, AvgT, Rank, total_reward = 0, 0, 0, 0, 0, 0
    SR_turn_15 = [0] * args.max_turn
    turn_result = []
    result = []
    user_size = test_env.ui_array.shape[0]
    print('User size in UI_test: ', user_size)
    test_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    plot_filename = 'Evaluate-{}'.format(i_episode) + filename
    if args.data_name in [LAST_FM_STAR, LAST_FM]:
        if args.eval_num == 1:
            test_size = 500
        else:
            test_size = 500  # Only do 500 iteration for the sake of time
        user_size = test_size
    if args.data_name in [YELP_STAR, YELP, INS]:
        test_size = 500  # XXX: To Fix

        user_size = test_size
    print('The select Test size : ', test_size)
    for user_num in tqdm(range(user_size)):  # user_size
        # TODO uncommend this line to print the dialog process
        # blockPrint()
        print('\n================test tuple:{}\
                ===================='.format(user_num))
        if not args.fix_emb:
            scpr, state, cand_feature, cand_item, \
                ac_label, action_space = test_env.reset(
                    agent.gcn_net.embedding.weight.data.cpu().detach().numpy())


# Reset environment and record the starting state
        else:
            scpr, state, cand_feature, cand_item, ac_label, action_space = test_env.reset(
            )
        epi_reward = 0
        is_last_turn = False
        for t in count():  # user  dialog
            if t == 14:
                is_last_turn = True
            action, sorted_actions, is_sample = agent.select_action(
                scpr,
                state,
                cand_feature,
                cand_item,
                action_space,
                is_test=True,
                is_last_turn=is_last_turn,
                turn=t)
            next_scpr, next_state, next_cand_feature, next_cand_item, nac_label, action_space, \
                reward, done = test_env.step(action.item(), sorted_actions)
            epi_reward += reward
            reward = torch.tensor([reward],
                                  device=args.device,
                                  dtype=torch.float)
            if done:
                next_state = None
            state = next_state
            scpr = next_scpr
            cand_feature = next_cand_feature
            cand_item = next_cand_item
            if done:
                enablePrint()
                if reward.item() == 1:  # recommend successfully
                    SR_turn_15 = [
                        v + 1 if i > t else v for i, v in enumerate(SR_turn_15)
                    ]
                    # for i, v in enumerate(SR_turn_15):
                    #     if i > t:
                    #         SR_turn_15 = v + 1
                    #     else:
                    #         SR_turn_15 = v
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
                total_reward += epi_reward
                AvgT += t + 1
                break
        if (user_num + 1) % args.observe_num == 0 and user_num > 0:
            SR = [
                SR5 / args.observe_num, SR10 / args.observe_num,
                SR15 / args.observe_num, AvgT / args.observe_num,
                Rank / args.observe_num, total_reward / args.observe_num
            ]
            SR_TURN = [i / args.observe_num for i in SR_turn_15]
            print('Total evalueation epoch_uesr:{}'.format(user_num + 1))
            print('Takes {} seconds to finish {}% of \
                    this task'.format(str(time.time() - start),
                                      float(user_num) * 100 / user_size))
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, reward:{} '
                  'Total epoch_uesr:{}'.format(
                      SR5 / args.observe_num, SR10 / args.observe_num,
                      SR15 / args.observe_num, AvgT / args.observe_num,
                      Rank / args.observe_num, total_reward / args.observe_num,
                      user_num + 1))
            result.append(SR)
            turn_result.append(SR_TURN)
            SR5, SR10, SR15, AvgT, Rank, total_reward = 0, 0, 0, 0, 0, 0
            SR_turn_15 = [0] * args.max_turn
            tt = time.time()
        enablePrint()

    SR5_mean = np.mean(np.array([item[0] for item in result]))
    SR10_mean = np.mean(np.array([item[1] for item in result]))
    SR15_mean = np.mean(np.array([item[2] for item in result]))
    AvgT_mean = np.mean(np.array([item[3] for item in result]))
    Rank_mean = np.mean(np.array([item[4] for item in result]))
    reward_mean = np.mean(np.array([item[5] for item in result]))
    SR_all = [
        SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean, reward_mean
    ]
    save_rl_mtric(dataset=args.data_name,
                  filename=filename,
                  epoch=user_num,
                  SR=SR_all,
                  spend_time=time.time() - start,
                  mode='test')
    save_rl_mtric(dataset=args.data_name,
                  filename=test_filename,
                  epoch=user_num,
                  SR=SR_all,
                  spend_time=time.time() - start,
                  mode='test')  # save RL SR
    print('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = np.mean(np.array([item[i] for item in turn_result]))
    print('success turn:{}'.format(SRturn_all))
    print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, reward:{}'.format(
        SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean, reward_mean))
    PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('Training epoch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(user_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')
    PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + plot_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('{}\t{}\t{}\t{}\t{}\n'.format(i_episode, SR15_mean, AvgT_mean,
                                              Rank_mean, reward_mean))

    return SR15_mean
