"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0

使用策略梯度来实现小摇杆和小汽车游戏的历史

策略梯度查看网址：http://www.algorithmdog.com/rl-policy-gradient

策略梯度是动作概率为基础的一类强化学习实现

以下是描述了一个使用蒙特卡罗策略梯度的实现，该方法是基于一个状态和奖励序列：
s1,a1,r1,…..,sT,aT,rT
在第 t 时刻，我们让 gt=rt+γrt+1+... 等于 q(st,a) ，从而求解策略梯度优化问题
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")  #给定环境
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")    #给定的动作的数量
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")  #给定的动作值
        # fc1
        """
        tf.layers.dense参数：
            units：整数或长整数，输出空间的维数。
            activation：激活功能（可调用），将其设置为“None”以保持线性激活。
            use_bias：Boolean，表示该层是否使用偏差。
            kernel_initializer：权重矩阵的初始化函数；如果为None（默认），则使用tf.get_variable使用的默认初始化程序初始化权重。
            bias_initializer：偏置的初始化函数。
            kernel_regularizer：权重矩阵的正则化函数。
            bias_regularizer：正规函数的偏差。
            activity_regularizer：输出的正则化函数。
            kernel_constraint：由Optimizer更新后应用于内核的可选投影函数（例如，用于实现层权重的范数约束或值约束）。该函数必须将未投影的变量作为输入，并且必须返回投影变量（必须具有相同的形状）。在进行异步分布式训练时，使用约束是不安全的。
            bias_constraint：由Optimizer更新后应用于偏置的可选投影函数。
            trainable：Boolean，如果为True，还将变量添加到图集合GraphKeys.TRAINABLE_VARIABLES中（请参阅参考资料tf.Variable）。
            name：String，图层的名称；具有相同名称的图层将共享权重，但为了避免错误，在这种情况下，我们需要reuse=True。
            reuse：Boolean，是否以同一名称重用前一层的权重。
        """
        layer = tf.layers.dense(
            inputs=self.tf_obs,   #给定输入
            units=10,   #整数或长整数，输出空间的维数
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),   #权重矩阵的初始化函数
            bias_initializer=tf.constant_initializer(0.1),  #正则函数的偏差
            name='fc1'  #图层名称
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,  #输入层
            units=self.n_actions,   #输出是动作个数
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),  #权重矩阵的初始化函数
            bias_initializer=tf.constant_initializer(0.1),     #正则函数的偏差
            name='fc2'   #定义名称
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):  #sum((R-b)*log(p(s|a)))
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:  也可以使用下面的方法来实现梯度计算：sum(log(p(s|a)))
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)

            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss  概率*奖励  (p*r)/N

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        #获取所有权重
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})

        #选择动作,ravel函数是一个原nd的视图，但是会改变原值，flatten函数不会修改原nd
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)  #记录状态
        self.ep_as.append(a)   #记录动作
        self.ep_rs.append(r)   #记录奖励

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()   #给定折扣：蒙特卡罗策略梯度

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]  环境
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]   动作数量
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]    动作值
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)   #给定初始化的折扣reward
        running_add = 0

        #反转(从后往前的乘以折扣因子，表示是最新的放到前面)
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add  #记录该值

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)   # （R-b）给定平均值作为一个baseline: E[R()]，防止没有采样的好动作以为内reward少而忽略
        discounted_ep_rs /= np.std(discounted_ep_rs)     #给定标准差
        return discounted_ep_rs



