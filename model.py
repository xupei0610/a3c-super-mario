import tensorflow as tf
import numpy as np
import cv2

from ops import *

class LSTMNetwork():
    def preprocess(self, state):
        if state.shape[0] == self.state_shape[0] and state.shape[1] == self.state_shape[1]:
            cropped = state[32:-16, :, :]
        else:
            cropped = cv2.resize(state[32:-16, :, :], (self.state_shape[0], self.state_shape[1]))
        colored = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        return np.reshape(colored, (self.state_shape[0], self.state_shape[1], 1)) / 255.0 - 0.5


    def __init__(self, n_actions, trainable,
                 entropy_beta=None, ppo_eps=None):
        # self.state_shape = [224-32-16,256,1]
        self.state_shape = [84, 84, 1]

        self.n_actions = n_actions

        self.X = tf.placeholder(tf.float32, [None] + self.state_shape, name="game_frame")
        self.conv1, self.conv1_w, self.conv1_b = conv_layer("conv1", self.X,      
                                    self.state_shape[-1], 32, 8, 4, "VALID",
                                    trainable=trainable, with_bn=False)
        self.conv2, self.conv2_w, self.conv2_b = conv_layer("conv2", self.conv1,
                                    32, 64, 4, 2, "VALID",
                                    trainable=trainable, with_bn=False)
        self.conv3, self.conv3_w, self.conv3_b = conv_layer("conv3", self.conv2,
                                    64, 64, 3, 1, "VALID",
                                    trainable=trainable, with_bn=False)
        self.lstm, (self.lstm_state, self.lstm_state_in, self.step_size) = lstm_layer("lstm", self.conv3,
                                    self.conv3.get_shape()[1:].num_elements(), 512)
        with tf.variable_scope("actor"):
            logits, self.w5, self.b5 = fc_layer("fc1",  self.lstm,
                                    512, self.n_actions,
                                    trainable=trainable, with_elu=False, with_bn=False)    # a@1 
            self.policy = tf.nn.softmax(logits)
        with tf.variable_scope("critic"):
            V, self.w6, self.b6 = fc_layer("fc1", self.lstm,
                                    512, 1,
                                    trainable=trainable, with_elu=False, with_bn=False)   # 1@1 
            self.value = tf.reshape(V, [-1])

        if entropy_beta is not None:
            self.advantage  = tf.placeholder(tf.float32, [None], name="advantage")
            self.action = tf.placeholder(tf.int32, [None], name="action")
            self.reward = tf.placeholder(tf.float32, [None], name="reward")
            if ppo_eps is not None:
                self.policy_old = tf.placeholder(tf.float32, [None, n_actions], name="pi_old")
            with tf.name_scope("batch_size"):
                self.batch_size = tf.shape(self.X)[0]
            # actor loss
            with tf.variable_scope("actor_loss"):
                action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0)
                if ppo_eps is None:
                    log_pi_a = tf.log(tf.reduce_sum(tf.multiply(self.policy, action_one_hot), axis=1)+1e-10, name="log_pi_a")
                    self.actor_loss = tf.reduce_sum(tf.multiply(log_pi_a, self.advantage))
                else:
                    pi_a = tf.reduce_sum(tf.multiply(self.policy, action_one_hot), axis=1, name="pi_a")
                    pi_old_a = tf.reduce_sum(tf.multiply(self.policy_old, action_one_hot), axis=1, name="pi_old_a")
                    ratio = tf.div(pi_a, pi_old_a)
                    surrogate = tf.multiply(ratio, self.advantage, name="surrogate")
                    clipped = tf.multiply(tf.clip_by_value(ratio, 1-ppo_eps, 1+ppo_eps), self.advantage, name="clip")
                    self.actor_loss = tf.reduce_sum(tf.minimum(surrogate, clipped))
                self.actor_loss = - self.actor_loss
            # critic loss
            with tf.variable_scope("critic_loss"):
                self.critic_loss = tf.nn.l2_loss(self.reward - self.value)
                if ppo_eps is None:
                    self.critic_loss = 0.5 * self.critic_loss
            # entropy loss
            # Mnih's paper \sum policy * -log(policy)
            with tf.name_scope("policy_entropy"):
                self.policy_entropy = entropy_beta*(-tf.reduce_sum(self.policy*tf.log(self.policy+1e-10)))
            with tf.name_scope("loss"):
                self.loss = self.critic_loss + self.actor_loss - self.policy_entropy


class CNNQNetwork():
    def preprocess(self, state):
        if state.shape[0] == self.state_shape[0] and state.shape[1] == self.state_shape[1]:
            cropped = state[32:-16, :, :]
        else:
            cropped = cv2.resize(state[32:-16, :, :], (self.state_shape[0], self.state_shape[1]))
        if self.state_shape[2] == 1:
            colored = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        else:
            colored = cropped
        return np.reshape(colored, self.state_shape) / 255.0 - 0.5

    
    def __init__(self, n_actions, dueling, trainable=True, prioritized_memory=False, huber_loss=False):
        # self.state_shape = [224-32-16,256,1]
        self.state_shape = [128,128,1]
        self.n_actions = n_actions

        self.X = tf.placeholder(tf.float32, [None] + self.state_shape[0:-1] + [self.state_shape[-1]*4], name="game_frames")

        self.conv1, self.conv1_w, self.conv1_b = conv_layer("conv1", self.X,      
                                    self.state_shape[-1]*4, 32, 8, 4, "VALID",
                                    trainable=trainable, with_bn=False)
        self.conv2, self.conv2_w, self.conv2_b = conv_layer("conv2", self.conv1,
                                    32, 64, 4, 2, "VALID",
                                    trainable=trainable, with_bn=False)
        self.conv3, self.conv3_w, self.conv3_b = conv_layer("conv3", self.conv2,
                                    64, 64, 3, 1, "VALID",
                                    trainable=trainable, with_bn=False)
        if trainable:
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            midware = tf.nn.dropout(self.conv3, self.keep_prob)
        else:
            midware = self.conv3

        self.fc1, self.fc1_w, self.fc1_b = fc_layer("fc1", midware,
                                    self.conv3.get_shape()[1:].num_elements(), 512,
                                    trainable=trainable, with_bn=False)     # 512
        if dueling:
            with tf.variable_scope("value"):
                self.value, self.value_w, self.value_b = fc_layer("val_fc1", self.fc1,
                                        512, 1,
                                        trainable=trainable, with_elu=False, with_bn=False)     # 1
            with tf.variable_scope("advantage"):
                self.advantage_fc1, self.advantage_w, self.advantage_b = fc_layer("adv_fc1", self.fc1,
                                        512, self.n_actions,
                                        trainable=trainable, with_elu=False, with_bn=False)  # n
                self.advantage = self.advantage_fc1 - tf.reduce_mean(self.advantage_fc1, axis=1, keepdims=True)
            with tf.variable_scope("Q"):
                self.Q = self.value + self.advantage
        else:
            self.Q, self.fc2_w, self.fc2_b = fc_layer("fc2", self.fc1,
                                        512, self.n_actions,
                                        trainable=trainable, with_elu=False, with_bn=False)  # n

        if trainable:
            self.action = tf.placeholder(tf.int32, [None], name="action")
            self.v_target = tf.placeholder(tf.float32, [None], name="v_target")
            if huber_loss:
                self.reward_clip = tf.placeholder(tf.float32, name="reward_clip")
            if prioritized_memory:
                self.is_weights = tf.placeholder(tf.float32, [None], name="is_weights")

            with tf.name_scope("td_error"):
                with tf.name_scope("Q_t_a"):
                    action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0)
                    Q_t_a = tf.reduce_sum(tf.multiply(self.Q, action_one_hot), axis=1)
                td_err = self.v_target - Q_t_a
        
            with tf.name_scope("loss"):
                if huber_loss:
                    with tf.name_scope("mse"):
                        mse = 0.5*tf.square(td_err)
                    with tf.name_scope("abs_err"):
                        self.abs_err = tf.abs(td_err)
                    with tf.name_scope("huber_loss"):
                        loss = tf.where(self.abs_err <= self.reward_clip,
                                    mse, self.reward_clip*self.abs_err - 0.5*tf.square(self.reward_clip))
                else:
                    with tf.name_scope("mse"):
                        loss = tf.square(td_err)

                if prioritized_memory:
                    self.loss = tf.reduce_mean(self.is_weights * loss)
                else:
                    self.loss = tf.reduce_mean(loss)
                