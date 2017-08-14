import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
from agent import *


def environment_info(env):
    ''' Prints info about the given environment. '''
    print('Observation space: {}'.format(env.observation_space))
    print('Observation space high values: {}'.format(env.observation_space.high))
    print('Observation space low values: {}'.format(env.observation_space.low))
    print('Number of actions available: {}'.format(env.action_space))


def basic_guessing_policy(env, agent):
    ''' Execute random guessing policy. '''
    totals = []
    for episode in range(500):
        episode_rewards = 0
        obs = env.reset()
        # env.render()
        for step in range(1000):  # 1000 steps max
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            # env.render()
            if done:
                break
        totals.append(episode_rewards)

    print('Average: {}'.format(np.mean(totals)))
    print('Standard Deviation: {}'.format(np.std(totals)))
    print('Minimum: {}'.format(np.min(totals)))
    print('Maximum: {}'.format(np.max(totals)))


def random_guessing_policy(env, agent):
    ''' Execute random guessing policy. '''
    totals = []
    for episode in range(500):
        episode_rewards = 0
        obs = env.reset()
        # env.render()
        for step in range(1000):  # 1000 steps max
            action = agent.act()
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            # env.render()
            if done:
                break
        totals.append(episode_rewards)

    print('Average: {}'.format(np.mean(totals)))
    print('Standard Deviation: {}'.format(np.std(totals)))
    print('Minimum: {}'.format(np.min(totals)))
    print('Maximum: {}'.format(np.max(totals)))


def q_learning(env, agent, gamma=0.9):
    '''
    Implement Q-learning policy.

    Args:
        env: enviroment object.
        agent: Learning agent.
        gamma: discount factor.
    Returns:
        Something...
    '''
    # Start out with Q-table set to zero.
    # Agent initially doesn't know how many states there are...
    # so if a new state is found, then add a new column/row to Q-table
    # Q[state][action] = R[state][action] + (gamma * max(Q[next_state][all_actions])
    # Q[state][action] = (1 - alpha) * Q[state][action] + alpha * (next_reward + (gamma * max(Q[next_state][all_actions]))
    # Or... Q[state][action] += alpha(next_reward + (gamma * max(Q[next_state][all_actions])) - Q[state][action])
    valid_actions = [0, 1]
    tolerance = 0.01
    training = True
    training_totals = []
    testing_totals = []
    epsilon_hist = []
    for episode in range(500):
        episode_rewards = 0
        obs = env.reset()
        if agent.epsilon < tolerance:     # if epsilon is less than tolerance, testing begins
            agent.alpha = 0
            agent.epsilon = 0
            training = False
        agent.epsilon = agent.epsilon * 0.987                           # 98.7% of epsilon value
        for step in range(1000):        # 1000 steps max
            state = agent.create_state(obs)                 # Get state
            agent.create_Q(state, valid_actions)            # Create state in Q_table
            action = agent.choose_action(state)             # Choose a random action when learning
            obs, reward, done, info = env.step(action)      # Do action
            if done:
                reward = -1
            episode_rewards += reward                       # Receive reward
            agent.learn(state, action, reward)              # Update Q-table
            if done:
                break
        if training:
            training_totals.append(episode_rewards)
            agent.training_trials += 1
            epsilon_hist.append(agent.epsilon)
        else:
            testing_totals.append(episode_rewards)
            agent.testing_trials += 1
    return training_totals, testing_totals, epsilon_hist


def moving_average(data, window_size=10):
    ''' Calculate moving average with given window size.

    Args:
        data: List of floats.
        window_size: Integer size of moving window.
    Returns:
        List of rolling averages with given window size.
    '''
    sum_vec = np.cumsum(np.insert(data, 0, 0))
    moving_ave = (sum_vec[window_size:] - sum_vec[:-window_size]) / window_size
    return moving_ave


def display_stats(agent, training_totals, testing_totals, epsilon_hist):
    ''' Print and plot the various statistics from q-learning data.

    Args:
        agent: Agent containing variables useful for post analysis
        training_totals: List of training rewards per episode.
        testing_totals: List of testing rewards per episode.
        epsilon_hist: List of all epsilon values.
    '''
    print('******* Training Stats *********')
    print('Average: {}'.format(np.mean(training_totals)))
    print('Standard Deviation: {}'.format(np.std(training_totals)))
    print('Minimum: {}'.format(np.min(training_totals)))
    print('Maximum: {}'.format(np.max(training_totals)))
    print('Number of training episodes: {}'.format(agent.training_trials))
    print()
    print('******* Testing Stats *********')
    print('Average: {}'.format(np.mean(testing_totals)))
    print('Standard Deviation: {}'.format(np.std(testing_totals)))
    print('Minimum: {}'.format(np.min(testing_totals)))
    print('Maximum: {}'.format(np.max(testing_totals)))
    print('Number of testing episodes: {}'.format(agent.testing_trials))
    fig = plt.figure(figsize=(10, 7))
    # Plot Parameters plot
    ax1 = fig.add_subplot(311)
    ax1.plot([num + 1 for num in range(agent.training_trials)], epsilon_hist, color='b', label='Exploration Factor (Epsilon)')
    ax1.plot([num + 1 for num in range(agent.training_trials)], [0.1] * agent.training_trials, color='r', label='Learning Factor (Alpha)')
    ax1.set(title='Paramaters Plot',
            ylabel='Parameter values',
            xlabel='Trials')
    # Plot rewards
    ax2 = fig.add_subplot(312)
    ax2.plot([num + 1 for num in range(agent.training_trials)], training_totals, color='m', label='Training', alpha=0.4, linewidth=2.0)

    ax2.plot([num + 1 for num in range(agent.testing_trials)], testing_totals, color='k', label='Testing', linewidth=2.0)
    ax2.set(title='Reward per trial',
            ylabel='Rewards',
            xlabel='Trials')
    # Plot rolling average rewards
    ax3 = fig.add_subplot(313)
    window_size = 10
    train_ma = moving_average(training_totals, window_size=window_size)
    ax3.plot([num + 1 for num in range(agent.training_trials - (window_size - 1))], train_ma, color='m', label='Training', alpha=0.4, linewidth=2.0)
    test_ma = moving_average(testing_totals, window_size=window_size)
    ax3.plot([num + 1 for num in range(agent.testing_trials - (window_size - 1))], test_ma, color='k', label='Testing', linewidth=2.0)
    ax3.set(title='Rolling Average Rewards',
            ylabel='Reward',
            xlabel='Trials')

    fig.subplots_adjust(hspace=0.3)
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    return


def save_info(agent, training_totals, testing_totals):
    ''' Write statistics into text file.

    Args:
        agent: Agent containing variables useful for post analysis
        training_totals: List of training rewards per episode.
        testing_totals: List of testing rewards per episode.
    '''
    with open('CartPole-v0_stats.txt', 'w') as file_obj:
        file_obj.write('/-------- Q-Learning --------\\\n')
        file_obj.write('\n/---- Training Stats ----\\\n')
        file_obj.write('Average: {}\n'.format(np.mean(training_totals)))
        file_obj.write('Standard Deviation: {}\n'.format(np.std(training_totals)))
        file_obj.write('Minimum: {}\n'.format(np.min(training_totals)))
        file_obj.write('Maximum: {}\n'.format(np.max(training_totals)))
        file_obj.write('Number of training episodes: {}\n'.format(agent.training_trials))
        file_obj.write('\n/---- Testing Stats ----\\\n')
        file_obj.write('Average: {}\n'.format(np.mean(testing_totals)))
        file_obj.write('Standard Deviation: {}\n'.format(np.std(testing_totals)))
        file_obj.write('Minimum: {}\n'.format(np.min(testing_totals)))
        file_obj.write('Maximum: {}\n'.format(np.max(testing_totals)))
        file_obj.write('Number of testing episodes: {}\n'.format(agent.testing_trials))
        file_obj.write('\n/---- Q-Table ----\\')
        for state in agent.Q_table:
            file_obj.write('\n State: ' + str(state) + '\n\tAction: ' + str(agent.Q_table[state]))
    # Save figure and display plot
    plt.savefig('plots.png')
    plt.show()
    return


def main():
    ''' Execute main program. '''
    # Create a cartpole environment
    # Observation: [horizontal position, velocity, angle of pole, angular velocity]
    # Rewards: +1 at every step. i.e. goal is to stay alive
    env = gym.make('CartPole-v0')
    environment_info(env)
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent', help='define the type of agent you want.')
    args = parser.parse_args()
    # Basic agent enabled
    if args.agent == 'basic':
        agent = AgentBasic()
        basic_guessing_policy(env, agent)
    # Random agent enabled
    elif args.agent == 'random':
        agent = AgentRandom(env.action_space)
        random_guessing_policy(env, agent)
    # Q-learning agent enabled
    elif args.agent == 'q-learning':
        agent = AgentLearning(env, alpha=0.1, epsilon=1.0)
        training_totals, testing_totals, epsilon_hist = q_learning(env, agent)
        display_stats(agent, training_totals, testing_totals, epsilon_hist)
        save_info(agent, training_totals, testing_totals)
    # No argument passed, agent defaults to Basic
    else:
        agent = AgentBasic()
        basic_guessing_policy(env, agent)


if __name__ == '__main__':
    main()
