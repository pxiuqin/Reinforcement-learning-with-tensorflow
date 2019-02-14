"""
This part of code is the Dyna-Q learning brain, which is a brain of the agent.
All decisions and learning processes are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

主要是使用Q-Learning来解决连续动作，
使用针对Actor-Critic中Critic中给定Actor一个更好的选择，而不是只告诉好和不好
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)  #使用一个dataframe来学习存储状态和动作的转换

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection  增加探索性
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            #随机打乱动作，然后从中选择一个
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal  选择最大的动作
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

#定义了一个环境模型（给定状态和动作给出奖励和下一个状态）
class EnvModel:
    """Similar to the memory buffer in DQN, you can store past experiences in here.
    Alternatively, the model can generate next state and reward signal accurately."""
    def __init__(self, actions):
        # the simplest case is to think about the model is a memory which has all past transition information
        self.actions = actions
        self.database = pd.DataFrame(columns=actions, dtype=np.object)

    #类似于经验回放，把转换过程存储起来
    def store_transition(self, s, a, r, s_):
        if s not in self.database.index:
            self.database = self.database.append(
                pd.Series(
                    [None] * len(self.actions),
                    index=self.database.columns,
                    name=s,
                ))
        self.database.set_value(s, a, (r, s_))

    def sample_s_a(self):
        s = np.random.choice(self.database.index) #随机给定一种状态
        a = np.random.choice(self.database.ix[s].dropna().index)    # filter out the None value
        return s, a

    def get_r_s_(self, s, a):
        r, s_ = self.database.ix[s, a]
        return r, s_
