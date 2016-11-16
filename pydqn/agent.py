#!/usr/bin/env python
# --------------------------------------------------------
# DQN in pycaffe
# Copyright (c) 2016 SLXrobot
# Written by Chao Yu
# --------------------------------------------------------


import os
import time
import random
import numpy as np
from tqdm import tqdm
import config
import caffe

from collections import deque



class Agent(object):
    def __init__(self, config, environment):
        self.config = config
        self.env = environment
        self.his_state = np.zeros([self.config.frame_size, self.config.screen_height, self.config.screen_width], dtype=np.float32)
        self.cur_state = np.zeros([self.config.frame_size, self.config.screen_height, self.config.screen_width], dtype=np.float32)
        self.memory = deque(maxlen = self.config.memory_size)
        self.num_game = 0
        self.min_reward = self.config.min_reward
        self.max_reward = self.config.max_reward
        self.train_frequency = self.config.train_frequency
        self.batch_size = self.config.batch_size
        self.frame_size = self.config.frame_size
        self.num_action = self.env.action_size
        self.reward = 0
        self.action = 0
        self.terminal = 0
        self.step = 0
        self.learn_start = self.config.learn_start
        self.solver_dir = self.config.root_dir + self.config.solver_dir
        self.deploy_dir = self.config.root_dir + self.config.deploy_dir
        self.model_dir = self.config.root_dir + self.config.model_dir


    def process(self, screen, reward, action, terminal):
        reward = max(self.min_reward, min(self.max_reward, reward))

        self.his_state = self.cur_state

        self.cur_state[1:] = self.cur_state[:-1]
        self.cur_state[0] = screen

        self.memory.append((self.his_state, action, reward, self.cur_state, terminal))

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.update_target_q_network()


    def update_target_q_network(self):
        self.target_q_net.blobs['frames'].reshape(self.batch_size, self.frame_size, self.config.screen_height, self.config.screen_width)
        self.target_q_net.blobs['target'].reshape(self.batch_size, self.num_action)
        self.target_q_net.blobs['filter'].reshape(self.batch_size, self.num_action)
        minibatch = random.sample(self.memory, self.batch_size)
        state_batch = np.zeros([self.batch_size, self.config.frame_size, self.config.screen_height, self.config.screen_width], dtype=np.float32)
        nextState_batch = np.zeros([self.batch_size, self.config.frame_size, self.config.screen_height, self.config.screen_width], dtype=np.float32)
        for i in range(self.batch_size):
            state_batch[i, ...] = minibatch[i][0]
            nextState_batch[i, ...] = minibatch[i][3]
        # get qvalue :batchsize * 1
        self.target_q_net.blobs['frames'].data[...] = nextState_batch
        self.target_q_net.forward()
        qvalue_batch = self.target_q_net.blobs['q_values'].data

        target_batch = np.zeros([self.batch_size, self.num_action])
        filter_batch = np.zeros([self.batch_size, self.num_action])
        for i in range(0, self.batch_size):
            filter_batch[i, minibatch[i][1]] = 1
            if minibatch[i][4]:
                target_batch[i, minibatch[i][1]] = minibatch[i][2]
            else:
                target_batch[i, minibatch[i][1]] = minibatch[i][2] + self.config.gama*np.max(qvalue_batch[i, :])

        self.target_q_net.blobs['frames'].data[...] = state_batch
        self.target_q_net.blobs['target'].data[...] = target_batch
        self.target_q_net.blobs['filter'].data[...] = filter_batch
        self.target_q_solver.step(1)
        if self.step % 500 == 0:
            self.target_q_solver.snapshot()
        return


    def predict(self):
        self.epsilon = self.config.ep_end + (self.config.ep_start - self.config.ep_end)*(self.step + 0.0)/self.config.ep_end_t
        self.epsilon = max(0.1, min(self.epsilon, 1.0))
        # choose an action randomly
        if random.random() < self.epsilon:
            action = random.randrange(self.env.action_size)
        # caculate the action acording to the estimate net
        else:
            self.target_q_net.blobs['frames'].reshape(1, self.frame_size, self.config.screen_height, self.config.screen_width)
            self.target_q_net.blobs['target'].reshape(1, self.num_action)
            self.target_q_net.blobs['filter'].reshape(1, self.num_action)
            self.target_q_net.blobs['frames'].data[...] = self.cur_state
            self.target_q_net.forward()
            action = np.argmax(self.target_q_net.blobs['q_values'].data[0])
            print self.target_q_net.blobs['q_values'].data[0]
        return action

    def train(self):
        self.target_q_solver = caffe.SGDSolver(self.solver_dir)
        self.target_q_net = self.target_q_solver.net
        print "train"
        screen, reward, action, terminal = self.env.new_random_game()

        # init replay memory
        ep_reward = 0
        ep_rewards = []
        for self.step in tqdm(range(0, self.config.max_step), ncols=70, initial=0):
            # 1. predict
            action = self.predict()
            # 2. act
            screen, reward, terminal = self.env.act(action, is_training=True)
            # 3. observe
            self.process(screen, reward, action, terminal)
            if terminal:
                screen, reward, action, terminal = self.env.new_random_game()
                self.num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

    def play(self):
        self.target_q_net = caffe.Net(self.deploy_dir, self.model_dir, caffe.TEST)
        screen, reward, action, terminal = self.env.new_random_game()
        current_reward = 0
        self.epsilon = self.config.test_ep
        for i in range(self.frame_size):
            self.cur_state[i, ...] = screen

        for t in tqdm(range(self.config.test_step_size), ncols=70):
            action = self.predict()
            screen, reward, terminal = self.env.act(action, is_training=False)
            self.cur_state[1:] = self.cur_state[:-1]
            self.cur_state[0] = screen

            current_reward += reward
            if terminal:
                break
        print "reward of this epoch is ", current_reward





