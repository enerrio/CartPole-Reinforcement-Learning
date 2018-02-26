# CartPole-Reinforcement-Learning
Reinforcement learning approach to OpenAI Gym's CartPole environment


## Description
The cartpole problem is an inverted pendelum problem where a stick is balanced upright on a cart. The cart can be moved left or right to and the goal is to keep the stick from falling over. A positive reward of +1 is received for every time step that the stick is upright. When it falls past a certain degree then the "episode" is over and a new one can begin. CartPole can be found in [OpenAI Gym's](https://gym.openai.com) list of trainable environments.
This project requires **Python 3.5** with the [Gym](https://gym.openai.com/docs), [numpy, and matplotlib](https://scipy.org/install.html)  libraries installed.
The CartPole-V0 environment is used.

In 'agent.py', there are 3 agent classes defined, each with a different algorithm attached. The basic agent simply moves the cart left if the stick is leaning to the left and moves the cart right if the stick is leaning to the right. The random agent randomly chooses an action (left or right) for the cart to move at every time step. The Q-Learning agent uses a [Q-Learning](https://en.wikipedia.org/wiki/Q-learning) algorithm to choose the best action given the current observation of the cartpole.

When using Q-learning, 'stats.py' plots the agent parameters (alpha & epsilon), rewards per trial, and rolling average rewards per trial. This is useful for visualizing how well the algorithm is performing.

### Usage

In a terminal or command window, navigate to the project directory `CartPole-Reinforcement-Learning/` (that contains this README) and run one of the following commands:

```python3 cartpole.py -a basic```

```python3 cartpole.py -a random```

```python3 cartpole.py -a q-learning```

```python3 cartpole.py --help ```

This will run the `cartpole.py` file with the given agent. Leaving the -a flag empty defaults to a basic agent.
