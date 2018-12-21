"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.

sarsa的一种提速手段
其实就是在单步sarsa和回合sarsa之间定义了许多的值
"""

from maze_env import Maze
from RL_brain import SarsaLambdaTable


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        # initial all zero eligibility trace
        RL.eligibility_trace *= 0  # 这里表示每一个回合开始要重新给定衰减矩阵E为0

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)  # 获取当前动作后的环境和reward情况

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))  # 根据环境选择下一个动作

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
