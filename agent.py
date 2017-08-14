import random


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
    def __init__(self, env, alpha=0.1, epsilon=1.0):
        self.env = env
        self.alpha = alpha          # learning factor
        self.epsilon = epsilon
        self.Q_table = dict()
        # Following variables for statistics
        self.training_trials = 0
        self.testing_trials = 0

    def create_state(self, obs):
        ''' Create state variable from observation.

        Args:
            obs: Observation list.
        Returns:
            state: State tuple
        '''
        state = tuple([state < 0.0 for state in obs])
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
            max_Q = self.get_maxQ(state)                    # Find max Q value
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

    def learn(self, state, action, reward):
        ''' Update the Q-values '''
        # Q_table[state][action] = (1 - alpha) * Q_table[state][action] + (alpha * (reward + (gamma * max(Q_table[next_state][all_actions]))))
        # Disregarding gamma for now...
        self.Q_table[state][action] = (1 - self.alpha) * self.Q_table[state][action] + (self.alpha * reward)
        return
