import os
import sys
import time
import random
import multiprocessing
from collections import deque
from itertools import islice

import tensorflow as tf
import numpy as np
import cv2

import model


class Worker(object):

    def __init__(self, name, task_id,
                entropy_beta, gamma, gae_gamma,
                log_folder, log_interval, sync_interval,
                n_workers, game_wrapper):
        self.frame_stack = 1
        self.task_id = task_id
        self.name = name + "_" + str(self.task_id)
        self.game = game_wrapper(self.name)
        self.gamma = gamma
        self.gae_gamma = gae_gamma
        self.log_interval = log_interval
        self.sync_interval = sync_interval
        n_actions = len(self.game.ACTION_MAP)
        with tf.variable_scope("global_net"):
            model.LSTMNetwork(n_actions, True)
            master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "global_net")
            self.global_step = tf.train.get_or_create_global_step()
            self.inc_global_step = tf.assign_add(self.global_step, 1)

        self.current_lstm_state = (0, 0)
        self.current_sync_lstm_state = (0, 0)
        self.writer = tf.summary.FileWriter(os.path.join(log_folder, self.name))
        
        if task_id == 0:
            for i in range(n_workers):
                if i != task_id:
                    net_name = name + "_" + str(i)
                    with tf.variable_scope(net_name):
                        local_net = model.LSTMNetwork(n_actions, True, entropy_beta=entropy_beta)
                        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net_name)
                        with tf.variable_scope("step"):
                            local_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="step")
                        with tf.variable_scope("game_round"):
                            game_round = tf.Variable(0, dtype=tf.int32, trainable=False, name="game_round")
                            tf.assign_add(game_round, 1)
                        # for v in master_vars:
                        #     print(v.name)
                        # print("\n")
                        # for v in local_vars:
                        #     print(v.name)
                    with tf.name_scope("sync_" + net_name):
                        pull_weights = [tf.assign(local_var, master_var) for local_var, master_var in zip(local_vars, master_vars)]
                        clipped_gradients, _ = tf.clip_by_global_norm(tf.gradients(local_net.loss, local_vars), 40.0)
                        tf.train.AdamOptimizer(1e-4).apply_gradients(zip(clipped_gradients, master_vars), global_step=local_step)

        with tf.variable_scope(self.name):
            self.local_net = model.LSTMNetwork(n_actions, True, entropy_beta=entropy_beta)
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            with tf.variable_scope("step"):
                self.local_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="step")
            with tf.variable_scope("game_round"):
                self.game_round = tf.Variable(0, dtype=tf.int32, trainable=False, name="game_round")
                inc_game_round = tf.assign_add(self.game_round, 1)
            # for v in master_vars:
            #     print(v.name)
            # print("\n")
            # for v in local_vars:
            #     print(v.name)
        with tf.name_scope("sync_" + self.name):
            pull_weights = [tf.assign(local_var, master_var) for local_var, master_var in zip(local_vars, master_vars)]
            clipped_gradients, _ = tf.clip_by_global_norm(tf.gradients(self.local_net.loss, local_vars), 40.0)
            self.push_weights = tf.train.AdamOptimizer(1e-4).apply_gradients(zip(clipped_gradients, master_vars), global_step=self.local_step)

        self.pull_weights = lambda step_context: step_context.session.run(pull_weights)
        self.inc_game_round = lambda step_context: step_context.session.run(inc_game_round)


    def get_val(self, frame):
        def _get_val(step_context):
            return step_context.session.run([self.local_net.value],
                feed_dict={ self.local_net.X: np.expand_dims(frame, axis=0),
                            self.local_net.lstm_state_in[0]: self.current_lstm_state[0],
                            self.local_net.lstm_state_in[1]: self.current_lstm_state[1],
                            self.local_net.step_size: [1]
            })[0][0]
        return _get_val
    

    def get_pi_val(self, frame):
        def _get_pi_val(step_context):
            pi, v, self.current_lstm_state = step_context.session.run(
                        [self.local_net.policy, self.local_net.value, self.local_net.lstm_state],
                        feed_dict={self.local_net.X: np.expand_dims(frame, axis=0),
                                    self.local_net.lstm_state_in[0]: self.current_lstm_state[0],
                                    self.local_net.lstm_state_in[1]: self.current_lstm_state[1],
                                    self.local_net.step_size: [1]}
                )
            return pi[0], v[0]
        return _get_pi_val


    def train(self, sess):
        # init cache buffer
        self.frame_cache = deque()
        self.reward_cache = deque()
        self.action_cache = deque()
        self.value_cache = deque()
        # init game environment
        self.game.init()

        self.sess = sess
        # sync from the master net
        self.sess.run_step_fn(self.pull_weights)
        while not self.sess.should_stop():
            # reset LSTM memory
            self.current_lstm_state = (np.zeros(self.local_net.lstm_state_in[0].shape), np.zeros(self.local_net.lstm_state_in[1].shape))
            self.sync_lstm_state = (np.zeros(self.local_net.lstm_state_in[0].shape), np.zeros(self.local_net.lstm_state_in[1].shape))
            # reset level
            self.game.set_level(0)
            # get initial state
            frame, reward, over = self.game.step(self.game.ACTION_MAP[random.randrange(len(self.game.ACTION_MAP))])
            frame = self.local_net.preprocess(frame)
            input_frames = frame
            for _ in range(self.frame_stack-1):
                input_frames = np.concatenate((input_frames, frame), axis=2)
            step = 0
            total_reward = 0
            while not over:
                step += 1

                self.frame_cache.append(input_frames)    # s(t)

                pi, v = sess.run_step_fn(self.get_pi_val(input_frames))
                a = self.action(pi)

                frame, reward, over = self.game.step(self.game.ACTION_MAP[a])
                input_frames = np.append(self.local_net.preprocess(frame), input_frames[:, :, :self.frame_stack-1], axis=2)
            
                # if self.task_id == 0:
                #     cv2.imwrite("screenshots/{}.png".format(step), (np.reshape(input_frames, (128, 128))+0.5)*255.0)
                #     print(reward, pi)

                # if self.name == "worker_0" and a > 2 and a < 11:
                    # print(reward, a > 2 and a < 11, sum(pi[3:7]), sum(pi[7:11]))
                    # print(pi)

                self.value_cache.append(v)        # v(t)
                self.action_cache.append(a)       # a(t->t+1) 
                self.reward_cache.append(reward)  # r(t->t+1, a)

                total_reward += reward

                if over or step % self.sync_interval == 0:
                    _, _ = self.sync(over, input_frames)
                    
                    self.frame_cache.clear()
                    self.reward_cache.clear()
                    self.action_cache.clear()
                    self.value_cache.clear()

                    if over:
                        game_round = sess.run_step_fn(self.inc_game_round)                       
                        summary = tf.Summary(value=[
                            tf.Summary.Value(tag="perf/reward",   simple_value=total_reward),
                            tf.Summary.Value(tag="perf/score",    simple_value=self.game.score),
                            tf.Summary.Value(tag="perf/distance", simple_value=self.game.max_distance),
                        ])
                        self.writer.add_summary(summary, game_round)
                        print("Round:{}, Reward: {:.4f}, Score: {}, Distance: {}, Worker: {}".format(
                            game_round, total_reward, self.game.score, self.game.max_distance, self.name))


    def sync(self, over, s_end):
        batch_size = len(self.frame_cache)
        if over:
            value_t_end = 0
        else:
            value_t_end = self.sess.run_step_fn(self.get_val(s_end))
        
        discounted_r  = []
        r = value_t_end
        for reward in reversed(self.reward_cache):
            r = reward + self.gamma*r
            discounted_r.append(r)
        reward = list(reversed(discounted_r))

        # TD error: r(t->t+1) + gamma * v(t+1) - v(t)
        td_err = [r+self.gamma*v_t1-v_t for r, v_t, v_t1 in zip(
            self.reward_cache,
            self.value_cache,
            list(islice(self.value_cache, 1, batch_size)) + [value_t_end]
        )]
        # Advantage error: E[r(t->t+1) + gamma * v(t+1)] - v(t)
        advantage = []
        adv = 0
        for ad in reversed(td_err):
            adv = ad + self.gae_gamma * adv
            advantage.append(adv)
        advantage = list(reversed(advantage))

        loss, policy_entropy, actor_loss, critic_loss, self.sync_lstm_state, global_step, local_step, _ = self.sess.run(
            [self.local_net.loss, self.local_net.policy_entropy, self.local_net.actor_loss, self.local_net.critic_loss,
             self.local_net.lstm_state, self.inc_global_step, self.local_step, self.push_weights],
            feed_dict={ self.local_net.X: self.frame_cache,
                        self.local_net.action: self.action_cache,
                        self.local_net.reward: reward,
                        self.local_net.advantage: advantage,
                        self.local_net.lstm_state_in[0]: self.sync_lstm_state[0],
                        self.local_net.lstm_state_in[1]: self.sync_lstm_state[1],
                        self.local_net.step_size: [batch_size]
        })

        self.sess.run_step_fn(self.pull_weights)

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="loss/loss", simple_value=loss/batch_size),
            tf.Summary.Value(tag="loss/policy_entropy", simple_value=policy_entropy/batch_size),
            tf.Summary.Value(tag="loss/policy_loss", simple_value=actor_loss/batch_size),
            tf.Summary.Value(tag="loss/value_loss", simple_value=critic_loss/batch_size)
        ])

        self.writer.add_summary(summary, local_step)
        return global_step, loss

    @staticmethod
    def action(policy):
        return random.choices(range(len(policy)), weights=policy, k=1)[0]
    

def dispatch(name, task_id, distribution_scheme,
            entropy_beta, gamma, gae_gamma,
            log_folder, save_folder, log_interval, sync_interval,
            n_workers, game_wrapper):
    cluster = tf.train.ClusterSpec(distribution_scheme)
    server = tf.train.Server(cluster, job_name=name, task_index=task_id,
                             config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    if name == "ps":
        server.join()
    else:
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:" + str(task_id), cluster=cluster
        )):
            worker = Worker(name, task_id,
                            entropy_beta, gamma, gae_gamma,
                            log_folder, log_interval, sync_interval,
                            n_workers, game_wrapper)
        is_chief = (task_id == 0)

        class GraphHook(tf.train.SessionRunHook):
            def begin(self):
                self.writer = tf.summary.FileWriter(log_folder)
            def after_create_session(self, session, coord):
                self.writer.add_graph(session.graph)
        class LogHook(tf.train.SessionRunHook):
            def begin(self):
                self.writer = tf.summary.FileWriter(log_folder)
            def before_run(self, run_context):
                return tf.train.SessionRunArgs((
                    worker.global_step, worker.local_net.batch_size,
                    worker.local_net.loss, worker.local_net.actor_loss, worker.local_net.critic_loss, worker.local_net.policy_entropy
                ))
            def after_run(self, run_context, run_values):
                self.writer.add_summary(
                    tf.Summary(value=[
                            tf.Summary.Value(tag="overall_loss/loss", simple_value=run_values.results[2]/run_values.results[1]),
                            tf.Summary.Value(tag="overall_loss/policy_entropy", simple_value=run_values.results[5]/run_values.results[1]),
                            tf.Summary.Value(tag="overall_loss/policy_loss", simple_value=run_values.results[3]/run_values.results[1]),
                            tf.Summary.Value(tag="overall_loss/value_loss", simple_value=run_values.results[4]/run_values.results[1])
                        ]), run_values.results[0]
                )
                if run_values.results[0] % log_interval == 0:
                    print("Global Step: {}, Loss: {:.4f}; Worker: {}; PID: {}".format(
                        run_values.results[0], run_values.results[2]/run_values.results[1], worker.name, os.getpid()))

        chief_hooks = [GraphHook()]
        hooks = [tf.train.StopAtStepHook(num_steps=1e6),
                 LogHook(),
                 tf.train.NanTensorHook(worker.local_net.loss)]

        with tf.train.MonitoredTrainingSession(server.target, is_chief, save_folder,
                                               hooks=hooks, chief_only_hooks=chief_hooks,
                                               save_checkpoint_secs=600) as sess:
            worker.train(sess)


import config
import game_wrapper

def train():
    distributions = {
        "ps": [
            "localhost:" + str(config.A3C_START_PORT)
        ],
        "worker": []
    }
    for i in range(config.A3C_N_WORKERS):
        distributions["worker"].append("localhost:{}".format(config.A3C_START_PORT+i+1))
    args = (distributions,
            config.ENTROPY_BETA, config.GAMMA, config.GAE_GAMMA,
            config.A3C_LOG_FOLDER, config.A3C_SAVE_FOLDER, config.LOG_INTERVAL, config.A3C_SYNC_INTERVAL,
            config.A3C_N_WORKERS, lambda env_name: game_wrapper.Wrapper(env_name, config.FRAME_GAP))
    ps = multiprocessing.Process(target=dispatch, args=("ps", 0)+args)
    ps.start()
    workers = []
    for i in range(config.A3C_N_WORKERS):
        workers.append(multiprocessing.Process(target=dispatch, args=("worker", i)+args))
    for w in reversed(workers):
        w.start()
        time.sleep(1)
    print("=== Process ID List ===")
    print(ps.pid)
    for w in workers:
        print(w.pid)
    print("=======================")
    ps.join()


def test(restore_folder=None):
    game = game_wrapper.Wrapper("a3c-test", config.FRAME_GAP)
    # game = gym.make("meta-SuperMarioBros-v0")

    with tf.variable_scope("global_net"):
        net = model.LSTMNetwork(len(game_wrapper.Wrapper.ACTION_MAP), False)

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.global_variables_initializer())

        if restore_folder is not None:
            cp = tf.train.get_checkpoint_state(restore_folder)
            if cp and cp.model_checkpoint_path:
                tf.train.Saver().restore(sess, cp.model_checkpoint_path)
                print("Checkpoint Loaded", cp.model_checkpoint_path)

        game.init()

        while True:
            # reset LSTM memory
            current_lstm_state = (np.zeros(net.lstm_state_in[0].shape), np.zeros(net.lstm_state_in[1].shape))
            # reset level
            game.set_level(0)
            # get initial state
            frame, reward, over = game.step(game.ACTION_MAP[random.randrange(len(game.ACTION_MAP))])
            frame = net.preprocess(frame)
            step = 0
            while not over:
                step += 1
                pi, current_lstm_state = sess.run(
                        [net.policy, net.lstm_state],
                        feed_dict={ net.X: np.expand_dims(frame, axis=0),
                                    net.lstm_state_in[0]: current_lstm_state[0],
                                    net.lstm_state_in[1]: current_lstm_state[1],
                                    net.step_size: [1]}
                )
                pi = pi[0]

                pi_max = max(pi)
                cand = [i for i, x in enumerate(pi) if x == pi_max]
                if len(cand) == 1:
                    a = cand[0]
                else:
                    a = random.choice(cand)

                # a = random.choices(range(len(pi)), weights=pi, k=1)[0]

                frame, reward, over = game.step(game_wrapper.Wrapper.ACTION_MAP[a])
                frame = net.preprocess(frame)

                print("Step:{}; Action: {}; Reward: {}, ProbConf: {:4f}".format(
                    step, a, reward, pi[a]
                ))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A3C Implementation for Super Mario Bros.")
    parser.add_argument("-m", "--mode", default="test", help="train or test (default)")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else: # args.mode == "test":
        test(config.A3C_SAVE_FOLDER)

