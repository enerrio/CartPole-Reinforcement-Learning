#!/usr/bin/env python

import random
import pandas as pd
import numpy as np


class AgentBasic(object):
    ''' Simple agent class. '''
    def __init__(self):
        pass

    def act(self, obs):
        ''' Based on angle, return action. '''
        angle = obs[2]
        return 0 if angle < 0 else 1


class AgentRandom(object):
    ''' Random agent class. '''
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        ''' Agent randomly chooses an action. '''
        return self.action_space.sample()


class AgentLearning(object):
    '''' Agent that can learn via Q-learning. '''
    def __init__(self, env, alpha=0.1, epsilon=1.0, gamma=0.9):
        self.env = env
        self.alpha = alpha          # Learning factor
        self.epsilon = epsilon
        self.gamma = gamma          # Discount factor
        self.Q_table = dict()
        self._set_seed()
        # Following variables for statistics
        self.training_trials = 0
        self.testing_trials = 0

    def _set_seed(self):
        ''' Set random seeds for reproducibility. '''
        np.random.seed(21)
        random.seed(21)

    def build_state(self, features):
        ''' Build state by concatenating features (bins) into 4 digit int. '''
        return int("".join(map(lambda feature: str(int(feature)), features)))

    def create_state(self, obs):
        ''' Create state variable from observation.

        Args:
            obs: Observation list with format [horizontal position, velocity,
                 angle of pole, angular velocity].
        Returns:
            state: State tuple
        '''
        cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
        pole_angle_bins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
        cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
        angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]
        state = self.build_state([np.digitize(x=[obs[0]], bins=cart_position_bins)[0],
                                 np.digitize(x=[obs[1]], bins=pole_angle_bins)[0],
                                 np.digitize(x=[obs[2]], bins=cart_velocity_bins)[0],
                                 np.digitize(x=[obs[3]], bins=angle_rate_bins)[0]])
        return state

    def choose_action(self, state):
        ''' Given a state, choose an action.

        Args:
            state: State of the agent.
        Returns:
            action: Action that agent will take.
        '''
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            # Find max Q value
            max_Q = self.get_maxQ(state)
            actions = []
            for key, value in self.Q_table[state].items():
                if value == max_Q:
                    actions.append(key)
            if len(actions) != 0:
                action = random.choice(actions)
        return action

    def create_Q(self, state, valid_actions):
        ''' Update the Q table given a new state/action pair.

        Args:
            state: List of state booleans.
            valid_actions: List of valid actions for environment.
        '''
        if state not in self.Q_table:
            self.Q_table[state] = dict()
            for action in valid_actions:
                self.Q_table[state][action] = 0.0
        return

    def get_maxQ(self, state):
        ''' Find the maximum Q value in a given Q table.

        Args:
            Q_table: Q table dictionary.
            state: List of state booleans.
        Returns:
            maxQ: Maximum Q value for a given state.
        '''
        maxQ = max(self.Q_table[state].values())
        return maxQ

    def learn(self, state, action, prev_reward, prev_state, prev_action):
        ''' Update the Q-values

        Args:
            state: State at current time step.
            action: Action at current time step.
            prev_reward: Reward at previous time step.
            prev_state: State at previous time step.
            prev_action: Action at previous time step.
        '''
        # Updating previous state/action pair so I can use the 'future state'
        self.Q_table[prev_state][prev_action] = (1 - self.alpha) * \
            self.Q_table[prev_state][prev_action] + self.alpha * \
            (prev_reward + (self.gamma * self.get_maxQ(state)))
        return
