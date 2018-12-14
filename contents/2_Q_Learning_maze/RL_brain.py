"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd

#Q-learning强化学习
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy  #这里是给定了epsilon贪婪系数
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):   #选择一个动作
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:   #随机选择
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)  #选择一个最大动作的action
        else:
            # choose random action
            action = np.random.choice(self.actions)   #为了表示探索性，要能随机的落到其他动作上
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)  #添加新状态
        q_predict = self.q_table.loc[s, a]  #找到对应的Q(S,A)
        if s_ != 'terminal':
            #R + gamma * Q(S`,a)  S`表示下一状态，a表示最大Q值的动作
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal

        #Q(St,At)=Q(St,At)+lr* (R + gamma * Q(St+1,a) - Q(St,At))
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,    #这里的state是一个有带坐标的数据
                )
            )