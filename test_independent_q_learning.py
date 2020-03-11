import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils

from indepedent_DQN import test_select_action, hard_coded_policy, env, Tron_DQN, input_stack

def evaluate(policy_net_1, policy_net_2):
    player_1_rewards = []
    player_2_rewards = []
    team_rewards = []
    player_1_win = []
    player_2_win = []
    team_win = []

    for e in range(1000):
        # Initialize the environment and state
        env.reset()
        input_stack.__init__(env)
        prev_hard_coded_a = 1  # players init to up
        prev_hard_coded_b = 1  # players init to up
        print('Starting episode:', e)
        while True:
            # Select and perform an action
            action_1 = test_select_action(policy_net_1, input_stack, env, player_num=1)
            action_2 = test_select_action(policy_net_2, input_stack, env, player_num=2)
            hard_coded_a = hard_coded_policy(env.observation, np.argwhere(env.head_board == 3)[0], prev_hard_coded_a,
                                             env.config.board_shape, env.action_space, eps=env.config.hcp_eps)
            hard_coded_b = hard_coded_policy(env.observation, np.argwhere(env.head_board == 4)[0], prev_hard_coded_b,
                                             env.config.board_shape, env.action_space, eps=env.config.hcp_eps)

            prev_hard_coded_a = hard_coded_a
            prev_hard_coded_b = hard_coded_b
            next_observation, reward, done, dictionary = env.step(
                [action_1.item(), action_2.item(), hard_coded_a, hard_coded_b])

            input_stack.update(env)

            if done:
                player_1_rewards.append(reward[0])
                player_2_rewards.append(reward[1])
                team_rewards.append(reward[0] + reward[1])
                player_1_win.append(reward[0] > 0)
                player_2_win.append(reward[1] > 0)
                team_win.append((reward[0] > 0) or (reward[1] > 0))
                break

            env.render()

    stats = [np.mean(player_1_rewards), np.std(player_1_rewards), np.mean(player_2_rewards), np.std(player_2_rewards),
             np.mean(team_rewards), np.std(team_rewards), np.sum(player_1_win), np.sum(player_2_win), np.sum(team_win)]

    return stats

def evaluate_hard():
    player_1_rewards = []
    player_2_rewards = []
    team_rewards = []
    player_1_win = []
    player_2_win = []
    team_win = []

    for e in range(1000):
        # Initialize the environment and state
        env.reset()
        input_stack.__init__(env)
        prev_action_1 = 1
        prev_action_2 = 1
        prev_hard_coded_a = 1  # players init to up
        prev_hard_coded_b = 1  # players init to up
        print('Starting episode:', e)
        while True:
            # Select and perform an action
            action_1 = hard_coded_policy(env.observation, np.argwhere(env.head_board == 1)[0], prev_action_1,
                                             env.config.board_shape, env.action_space, eps=env.config.hcp_eps)
            action_2 = hard_coded_policy(env.observation, np.argwhere(env.head_board == 2)[0], prev_action_2,
                                             env.config.board_shape, env.action_space, eps=env.config.hcp_eps)
            hard_coded_a = hard_coded_policy(env.observation, np.argwhere(env.head_board == 3)[0], prev_hard_coded_a,
                                             env.config.board_shape, env.action_space, eps=env.config.hcp_eps)
            hard_coded_b = hard_coded_policy(env.observation, np.argwhere(env.head_board == 4)[0], prev_hard_coded_b,
                                             env.config.board_shape, env.action_space, eps=env.config.hcp_eps)

            prev_action_1 = action_1
            prev_action_2 = action_2
            prev_hard_coded_a = hard_coded_a
            prev_hard_coded_b = hard_coded_b
            next_observation, reward, done, dictionary = env.step(
                [action_1, action_2, hard_coded_a, hard_coded_b])

            if done:
                player_1_rewards.append(reward[0])
                player_2_rewards.append(reward[1])
                team_rewards.append(reward[0] + reward[1])
                player_1_win.append(reward[0] > 0)
                player_2_win.append(reward[1] > 0)
                team_win.append((reward[0] > 0) or (reward[1] > 0))
                break

            env.render()

    stats = [np.mean(player_1_rewards), np.std(player_1_rewards), np.mean(player_2_rewards), np.std(player_2_rewards),
             np.mean(team_rewards), np.std(team_rewards), np.sum(player_1_win), np.sum(player_2_win), np.sum(team_win)]

    return stats

def plot(stats_list):
    avg_reward_1 = np.array([stats[0] for stats in stats_list])
    std_reward_1 = np.array([stats[1] for stats in stats_list])
    avg_reward_2 = np.array([stats[2] for stats in stats_list])
    std_reward_2 = np.array([stats[3] for stats in stats_list])
    avg_reward_team = np.array([stats[4] for stats in stats_list])
    std_reward_team = np.array([stats[5] for stats in stats_list])

    num_wins_1 = np.array([stats[6] for stats in stats_list])
    num_wins_2 = np.array([stats[7] for stats in stats_list])
    num_wins_team = np.array([stats[8] for stats in stats_list])

    episode = np.arange(1, len(avg_reward_1) + 1)

    utils.cond_mkdir('C:/Users/Aruns/Documents/CS234/Project/project_ind/multi-agent_Tron/plots')

    plt.figure()
    plt.plot(episode, avg_reward_1)
    reward_upper = avg_reward_1 + std_reward_1
    reward_lower = avg_reward_1 - std_reward_1
    plt.fill_between(episode, reward_lower, reward_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlabel('Evaluation #')
    plt.ylabel('Reward')
    plt.title('Average Reward of Player 1')
    plt.savefig('C:/Users/Aruns/Documents/CS234/Project/project_ind/multi-agent_Tron/plots/reward_1')

    plt.figure()
    plt.plot(episode, avg_reward_2)
    reward_upper = avg_reward_2 + std_reward_2
    reward_lower = avg_reward_2 - std_reward_2
    plt.fill_between(episode, reward_lower, reward_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlabel('Evaluation #')
    plt.ylabel('Reward')
    plt.title('Average Reward of Player 2')
    plt.savefig('C:/Users/Aruns/Documents/CS234/Project/project_ind/multi-agent_Tron/plots/reward_2')

    plt.figure()
    plt.plot(episode, avg_reward_team)
    reward_upper = avg_reward_team + std_reward_team
    reward_lower = avg_reward_team - std_reward_team
    plt.fill_between(episode, reward_lower, reward_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlabel('Evaluation #')
    plt.ylabel('Reward')
    plt.title('Average Reward of Team')
    plt.savefig('C:/Users/Aruns/Documents/CS234/Project/project_ind/multi-agent_Tron/plots/reward_team')

    plt.figure()
    plt.plot(episode, 100 * num_wins_1 / env.config.EVAL_EPISODE)
    plt.xlabel('Evaluation #')
    plt.ylabel('Win (%)')
    plt.title('Win % of Player 1')
    plt.savefig('C:/Users/Aruns/Documents/CS234/Project/project_ind/multi-agent_Tron/plots/wins_1')

    plt.figure()
    plt.plot(episode, 100 * num_wins_2 / env.config.EVAL_EPISODE)
    plt.xlabel('Evaluation #')
    plt.ylabel('Win (%)')
    plt.title('Win % of Player 2')
    plt.savefig('C:/Users/Aruns/Documents/CS234/Project/project_ind/multi-agent_Tron/plots/wins_2')

    plt.figure()
    plt.plot(episode, 100 * num_wins_team / env.config.EVAL_EPISODE)
    plt.xlabel('Evaluation #')
    plt.ylabel('Win (%)')
    plt.title('Win % of Team')
    plt.savefig('C:/Users/Aruns/Documents/CS234/Project/project_ind/multi-agent_Tron/plots/wins_team')

model_dir = 'C:/Users/Aruns/Documents/CS234/Project/project_ind/multi-agent_Tron/models'

models = os.listdir(model_dir)
stats_list = []

models = models[-2:]

for i in range(0, len(models), 2):
    model_1 = models[i]
    model_2 = models[i+1]
    policy_net_1 = torch.load(os.path.join(model_dir, model_1), map_location=torch.device('cpu'))
    policy_net_2 = torch.load(os.path.join(model_dir, model_2), map_location=torch.device('cpu'))
    #stats_list.append(evaluate(policy_net_1, policy_net_2))
    stats_list.append(evaluate_hard())

#plot(stats_list)
print(stats_list)
    