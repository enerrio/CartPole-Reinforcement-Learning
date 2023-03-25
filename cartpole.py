#!/usr/bin/env python

import gym
import numpy as np
import argparse
from agent import AgentBasic, AgentRandom, AgentLearning
import stats


def environment_info(env):
    ''' Prints info about the given environment. '''
    print('************** Environment Info **************')
    print('Observation space: {}'.format(env.observation_space))
    print('Observation space high values: {}'.format(env.observation_space.high))
    print('Observation space low values: {}'.format(env.observation_space.low))
    print('Action space: {}'.format(env.action_space))
    print()


def basic_guessing_policy(env, agent):
    ''' Execute basic guessing policy. '''
    totals = []
    for episode in range(500):
        episode_rewards = 0
        obs = env.reset()
        # env.render()
        for step in range(1000):  # 1000 steps max unless failure
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            # env.render()
            if done:
                # Terminal state reached, reset environment
                break
        totals.append(episode_rewards)

    print('************** Reward Statistics **************')
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
                # Terminal state reached, reset environment
                break
        totals.append(episode_rewards)

    print('Average: {}'.format(np.mean(totals)))
    print('Standard Deviation: {}'.format(np.std(totals)))
    print('Minimum: {}'.format(np.min(totals)))
    print('Maximum: {}'.format(np.max(totals)))


def q_learning(env, agent):
    '''
    Implement Q-learning policy.

    Args:
        env: Gym enviroment object.
        agent: Learning agent.
    Returns:
        Rewards for training/testing and epsilon/alpha value history.
    '''
    # Start out with Q-table set to zero.
    # Agent initially doesn't know how many states there are...
    # so if a new state is found, then add a new column/row to Q-table
    valid_actions = [0, 1]
    tolerance = 0.001
    training = True
    training_totals = []
    testing_totals = []
    history = {'epsilon': [], 'alpha': []}
    for episode in range(800):  # 688 testing trials
        episode_rewards = 0
        obs = env.reset()
        # If epsilon is less than tolerance, testing begins
        if agent.epsilon < tolerance:
            agent.alpha = 0
            agent.epsilon = 0
            training = False
        # Decay epsilon as training goes on
        agent.epsilon = agent.epsilon * 0.99  # 99% of epsilon value
        for step in range(200):        # 200 steps max
            state = agent.create_state(obs)           # Get state
            agent.create_Q(state, valid_actions)      # Create state in Q_table
            action = agent.choose_action(state)         # Choose action
            obs, reward, done, info = env.step(action)  # Do action
            episode_rewards += reward                   # Receive reward
            # Skip learning for first step
            if step != 0:
                # Update Q-table
                agent.learn(state, action, prev_reward, prev_state, prev_action)
            prev_state = state
            prev_action = action
            prev_reward = reward
            if done:
                # Terminal state reached, reset environment
                break
        if training:
            training_totals.append(episode_rewards)
            agent.training_trials += 1
            history['epsilon'].append(agent.epsilon)
            history['alpha'].append(agent.alpha)
        else:
            testing_totals.append(episode_rewards)
            agent.testing_trials += 1
            # After 100 testing trials, break. Because of OpenAI's rules for solving env
            if agent.testing_trials == 100:
                break
    return training_totals, testing_totals, history


def main():
    ''' Execute main program. '''
    # Create a cartpole environment
    # Observation: [horizontal pos, velocity, angle of pole, angular velocity]
    # Rewards: +1 at every step. i.e. goal is to stay alive
    env = gym.make('CartPole-v0')
    # Set environment seed
    env.seed(21)
    environment_info(env)
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent', help='define type of agent you want')
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
        agent = AgentLearning(env, alpha=0.9, epsilon=1.0, gamma=0.9)
        training_totals, testing_totals, history = q_learning(env, agent)
        stats.display_stats(agent, training_totals, testing_totals, history)
        stats.save_info(agent, training_totals, testing_totals)
        # Check if environment is solved
        if np.mean(testing_totals) >= 195.0:
            print("Environment SOLVED!!!")
        else:
            print("Environment not solved.",
                  "Must get average reward of 195.0 or",
                  "greater for 100 consecutive trials.")
    # No argument passed, agent defaults to Basic
    else:
        agent = AgentBasic()
        basic_guessing_policy(env, agent)


if __name__ == '__main__':
    ''' Run main program. '''
    main()
