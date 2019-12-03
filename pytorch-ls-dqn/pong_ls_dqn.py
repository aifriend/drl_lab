#!/usr/bin/env python3
import copy
import random

import gym
import numpy as np
# import argparse
import torch
import torch.optim as optim
import utils.dqn_model as dqn_model
import utils.experience as experience
import utils.utils as utils
import utils.wrappers as wrappers
from tensorboardX import SummaryWriter
from utils.actions import EpsilonGreedyActionSelector
from utils.agent import DQNAgent, TargetNet
from utils.hyperparameters import HYPERPARAMS
from utils.srl_algorithms import ls_step, ls_step_dueling

if __name__ == "__main__":
    params = HYPERPARAMS['pong']
    #    params['epsilon_frames'] = 200000
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    #     args = parser.parse_args()
    #     device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # conditional_update:
    # if true, test the updated weights before replacing the old ones,
    # if the new weights perform better, then replace them (bool)
    conditional_update = False
    env = gym.make(params['env_name'])
    env = wrappers.wrap_dqn(env)
    if conditional_update:
        test_env = gym.make(params['env_name'])
        test_env = wrappers.wrap_dqn(test_env)
    training_random_seed = 10
    save_freq = 50000
    n_drl = 100000  # steps of DRL between SRL
    n_srl = params['replay_size']  # size of batch in SRL step
    num_srl_updates = 3  # number of to SRL updates to perform
    use_double_dqn = False
    use_dueling_dqn = True
    use_boosting = False
    use_ls_dqn = True
    use_constant_seed = False  # to compare performance independently of the randomness
    save_for_analysis = False  # save also the replay buffer for later analysis

    lam = 1  # regularization parameter
    params['batch_size'] = 64
    if use_ls_dqn:
        print("using ls-dqn with lambda:", str(lam))
        model_name = "-LSDQN-LAM-" + str(lam) + "-" + str(int(1.0 * n_drl / 1000)) + "K"
    else:
        model_name = "-DQN"
    model_name += "-BATCH-" + str(params['batch_size'])
    if use_double_dqn:
        print("using double-dqn")
        model_name += "-DOUBLE"
    if use_dueling_dqn:
        print("using dueling-dqn")
        model_name += "-DUELING"
    if use_boosting:
        print("using boosting")
        model_name += "-BOOSTING"
    if conditional_update:
        print("using conditional update")
        model_name += "-COND"
    if use_constant_seed:
        model_name += "-SEED-" + str(training_random_seed)
        np.random.seed(training_random_seed)
        random.seed(training_random_seed)
        env.seed(training_random_seed)
        torch.manual_seed(training_random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_random_seed)
        print("training using constant seed of ", training_random_seed)

    writer = SummaryWriter(comment="-" + params['run_name'] + model_name)
    if use_dueling_dqn:
        net = dqn_model.DuelingLSDQN(env.observation_space.shape, env.action_space.n).to(device)
    else:
        net = dqn_model.LSDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = TargetNet(net)

    selector = EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = utils.EpsilonTracker(selector, params)

    agent = DQNAgent(net, selector, device=device)

    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])  # TODO: change to RMSprop

    utils.load_agent_state(net, optimizer, selector, load_optimizer=False)

    frame_idx = 0
    drl_updates = 0

    with utils.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    if save_for_analysis:
                        temp_model_name = model_name + "_" + str(frame_idx)
                        utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                                               selector.epsilon, save_replay=True, replay_buffer=buffer.buffer,
                                               name=temp_model_name)
                    else:
                        utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                                               selector.epsilon)
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = utils.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'],
                                         device=device, double_dqn=use_double_dqn)
            loss_v.backward()
            optimizer.step()
            drl_updates += 1

            # LS-UPDATE STEP
            if use_ls_dqn and (drl_updates % n_drl == 0) and (len(buffer) >= n_srl):
                # if len(buffer) > 1:
                print("performing ls step...")
                batch = buffer.sample(n_srl)
                if use_dueling_dqn:
                    if conditional_update:
                        w_adv_last_dict_before = copy.deepcopy(net.fc2_adv.state_dict())
                        w_val_last_dict_before = copy.deepcopy(net.fc2_val.state_dict())
                    ls_step_dueling(net, tgt_net.target_model, batch, params['gamma'], len(buffer), lam=lam,
                                    m_batch_size=256,
                                    device=device,
                                    use_boosting=use_boosting, use_double_dqn=use_double_dqn)
                    if conditional_update:
                        print("comparing old and new weights...")
                        w_adv_last_dict_after = copy.deepcopy(net.fc2_adv.state_dict())
                        w_val_last_dict_after = copy.deepcopy(net.fc2_val.state_dict())
                        test_agent = copy.deepcopy(agent)
                        # test original
                        test_agent.dqn_model.fc2_adv.load_state_dict(w_adv_last_dict_before)
                        test_agent.dqn_model.fc2_val.load_state_dict(w_val_last_dict_before)
                        before_reward = utils.test_agent(test_env, test_agent)
                        # test new
                        test_agent.dqn_model.fc2_adv.load_state_dict(w_adv_last_dict_after)
                        test_agent.dqn_model.fc2_val.load_state_dict(w_val_last_dict_after)
                        after_reward = utils.test_agent(test_env, test_agent)
                        print("average reward:: original: %.3f" % before_reward, " least-squares: %.3f" % after_reward)
                        if after_reward > before_reward:
                            net.fc2_adv.load_state_dict(w_adv_last_dict_after)
                            net.fc2_val.load_state_dict(w_val_last_dict_after)
                            print("using updated weights.")
                        else:
                            net.fc2_adv.load_state_dict(w_adv_last_dict_before)
                            net.fc2_val.load_state_dict(w_val_last_dict_before)
                            print("using original weights.")
                else:
                    if conditional_update:
                        w_last_before = copy.deepcopy(net.fc2.state_dict())
                    ls_step(net, tgt_net.target_model, batch, params['gamma'], len(buffer), lam=lam,
                            m_batch_size=256, device=device, use_boosting=use_boosting,
                            use_double_dqn=use_double_dqn)
                    if conditional_update:
                        print("comparing old and new weights...")
                        w_last_after = copy.deepcopy(net.fc2.state_dict())
                        test_agent = copy.deepcopy(agent)
                        # test original
                        test_agent.dqn_model.fc2.load_state_dict(w_last_before)
                        before_reward = utils.test_agent(test_env, test_agent)
                        # test new
                        agent.dqn_model.fc2.load_state_dict(w_last_after)
                        after_reward = utils.test_agent(test_env, test_agent)
                        print("average reward:: original: %.3f" % before_reward, " least-squares: %.3f" % after_reward)
                        if after_reward > before_reward:
                            net.fc2.load_state_dict(w_last_after)
                            print("using updated weights.")
                        else:
                            net.fc2.load_state_dict(w_last_before)
                            print("using original weights.")

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
            if frame_idx % save_freq == 0:
                if save_for_analysis and frame_idx % n_drl == 0:
                    temp_model_name = model_name + "_" + str(frame_idx)
                    utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                                           selector.epsilon, save_replay=True, replay_buffer=buffer.buffer,
                                           name=temp_model_name)
                else:
                    utils.save_agent_state(net, optimizer, frame_idx, len(reward_tracker.total_rewards),
                                           selector.epsilon)
