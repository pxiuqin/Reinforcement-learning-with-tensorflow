"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                    pd.Series(
                            [0] * len(self.actions),
                            index=self.q_table.columns,
                            name=state,
                    )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection  这里是关于epsilon-greedy来实现一个随机动作
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]  # 找到所有的动作信息
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)  # 找到最大值动作
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# backward eligibility traces
class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        # backward view, eligibility trace.
        self.lambda_ = trace_decay  # 定义一个衰减值
        self.eligibility_trace = self.q_table.copy()  # 复制一份原来的table定义成一个有效跟踪

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
            )
            self.q_table = self.q_table.append(to_be_append)

            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    """
    学习过程说明：
    1.创建一个矩阵E，用于记录每个回合中走过的路径和衰减情况（初始值为0）
    2.每执行一个(S,A)，对相应的E(S,A)进行一次更新（更新方式有两种）
    3.每一次Q更新是对整个Q矩阵进行更新，表示为：Q(St,At)=Q(St,At)+lr* (R + gamma * Q(St+1,At+1) - Q(St,At)) * E
    4.对E矩阵进行衰减，表示为：E=gamma * lambda * E
    5.每个回合结束要对E矩阵进行清0（因为E只是记录当前回合的路径）
    """

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            # R + gamma * Q(St+1,At+1)
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict  # 给定义损失情况

        # increase trace amount for visited state-action pair

        # Method 1:
        # accumulating trace：每次走到当前state，则将当前的eligibility_trace+1即
        # self.eligibility_trace.loc[s, a] += 1

        # Method 2:
        # replacing trace：给eligibility_trace设置上限，使得其所有值在（0，1）之间，所以每次更新时先将state所在的行清零，再将相应的E（S,A）E（S,A）置一，即：
        self.eligibility_trace.loc[s, :] *= 0  # 状态所在的行清0
        self.eligibility_trace.loc[s, a] = 1  # 将E[s,a]赋值为1

        # 注：accumulating trace方式没有上界，容易引起权重过大，所以一般选择replacing trace方式更好

        # Q update
        # Q(St,At)=Q(St,At)+lr* (R + gamma * Q(St+1,At+1) - Q(St,At)) * E
        self.q_table += self.lr * error * self.eligibility_trace

        # decay eligibility trace after update
        # E=gamma*lambda*E
        self.eligibility_trace *= self.gamma * self.lambda_

        # 这里没有清理矩阵E，在run_this.py中体现了
