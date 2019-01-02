"""
为什么会出现DQN
使用表格存储的Q-Learning算法中如果状态值多到表格无法记录，严重影响性能，
神经网络比较在行这种状态值多的问题，于是乎我们可以将状态和动作当成神经网络的输入,
然后经过神经网络分析后得到动作的 Q 值, 这样我们就没必要在表格中记录 Q 值, 而是直接使用神经网络生成 Q 值.
还有一种形式的是这样, 我们也能只输入状态值, 输出所有的动作值,
然后按照 Q learning 的原则, 直接选择拥有最大值的动作当做下一步要做的动作.

DQN有两大关键概念
记忆库：用于记录学习的经验
随机抽取：Fixed Q-targets 也是一种打乱相关性的机理

This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3

为了使用 Tensorflow 来实现 DQN, 比较推荐的方式是搭建两个神经网络,
target_net 用于预测 q_target 值,
他不会及时更新参数. eval_net 用于预测 q_eval, 这个神经网络拥有最新的神经网络参数.
不过这两个神经网络结构是完全一样的, 只是里面的参数不一样
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,  # 递增epsilon
            output_graph=False,
    ):
        self.n_actions = n_actions  # 传递动作
        self.n_features = n_features  # 传递过来的特征
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy  # epsilon的最大值
        self.replace_target_iter = replace_target_iter  # 更换 target_net 的步数，多少步后开始替换参数
        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size  # 每次更新时从 memory 里面取多少记忆出来
        self.epsilon_increment = e_greedy_increment  # epsilon的增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 给定epsilon-greedy值，是否开启探索模式, 并逐步减少探索次数

        # total learning step，记录学习次数 (用于判断是否更换 target_net 参数)
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]  初始化记忆，这里默认都是0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]  构建target_net和eval_net
        self._build_net()
        t_params = tf.get_collection('target_net_params')  # 提取target_net
        e_params = tf.get_collection('eval_net_params')  # 提取eval_net
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]  # 更新target_net参数

        self.sess = tf.Session()

        # 输出 tensorboard 文件
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []  # 记录所有 cost 变化, 用于最后 plot 出来观看

    """
    两个神经网络是为了固定住一个神经网络 (target_net) 的参数, target_net 是 eval_net 的一个历史版本,
    拥有 eval_net 很久之前的一组参数, 而且这组参数被固定一段时间, 然后再被 eval_net 的新参数所替换.
    而 eval_net 是不断在被提升的, 所以是一个可以被训练的网络 trainable=True. 而 target_net 的 trainable=False 不参与训练.
    """
    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        # input  给定了多行两列的状态输入，用来接收observation
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions],
                                       name='Q_target')  # for calculating loss，给定了多行4列的值，用来接收q_target的值
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net （定义第一层）
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input  给定的两列
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    # 存储转换的过程
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):  # 判断类是否有给定的属性
            self.memory_counter = 0

        # 记录一条转换记录[s,a,r,s_]的过程
        transition = np.hstack((s, [a, r], s_))  # hstack按照行对数据进行拼接的，然后组成一个大的tensor

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size  # 找到老的记忆值的位置
        self.memory[index, :] = transition  # 替换掉老的记忆值

        self.memory_counter += 1  # counter值累加

    # 选择动作
    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]  # 指定observation的形状

        # 给定一个随机情况，这里仍然使用epsilon-greedy来避免探索不足
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})  # 从eval_net中生成所有action的值
            action = np.argmax(actions_value)  # 找到最大的actions_value值
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters   检查是否替换target_net参数
        if self.learn_step_counter % self.replace_target_iter == 0:  # 指定了替换参数的步数
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory，从memory中随机抽取出指定大小的记忆值，这里注意不是全部，只是指定的一个范围
        if self.memory_counter > self.memory_size:  # 达到了记忆上限值
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]  # 存储到一个batch中

        # 获取 q_next (target_net 产生了 q) 和 q_eval(eval_net 产生的 q)
        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={
                    self.s_: batch_memory[:, -self.n_features:],  # fixed params
                    self.s: batch_memory[:, :self.n_features],  # newest params
                })

        # 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
        # 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
        # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
        # 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
        # q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.
        # q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
        # 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.

        # 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
        # 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
        # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
        # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.
        # 具体在下面还有一个举例说明.

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()   #全部赋值
        batch_index = np.arange(self.batch_size, dtype=np.int32)  #给定一个batch
        eval_act_index = batch_memory[:, self.n_features].astype(int)  #给定action的index，然后赋值
        reward = batch_memory[:, self.n_features + 1]  #给定reward的index，然后赋值
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)   #action的位置来给定新值，tensor中的值更新指定的内容

        """
        For example in this batch I have 2 samples and 3 actions:  #3个action中有2个batch
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.  #根据memory中的action值来更新q_target
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;  #这里要更新动作0的q值为-1
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]    #

        We then backpropagate this error w.r.t the corresponding action to network,  #差值作为误差，反向传播
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network 训练eval_net
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon，递增了epsilon值来减少探索性（随机性）
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
