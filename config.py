#!/usr/bin/env python
# --------------------------------------------------------
# DQN in pycaffe
# Copyright (c) 2016 SLXrobot
# Written by Chao Yu
# --------------------------------------------------------

import sys

class AgentConfig(object):

    scale = 10000
    display = True

    max_step = 5000 * scale
    memory_size = 10 * scale
    learn_start = 5. * scale

    batch_size = 32
    frame_size = 4

    target_q_update_step = 1 * scale
    train_frequency = 4

    num_action = 6
    gama = 0.7

    ep_end = 0.1
    ep_start = 1.
    ep_end_t = memory_size

    test_step_size = 100
    test_ep = 0.1




    random_start = 30
    cnn_format = 'NCHW'
    discount = 0.99
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5 * scale

    min_delta = -1
    max_delta = 1

    double_q = False
    dueling = False

    _test_step = 5 * scale
    _save_step = _test_step * 10

class EnvironmentConfig(object):
    env_name = 'Breakout-v0'
    screen_width = 84
    screen_height = 84
    max_reward = 1.
    min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
    backend = 'tf'
    env_type = 'detail'
    action_repeat = 1
    pycaffe_dir = "/media/foolyc/ssd/workspace/PycharmProjects/py-faster-rcnn-master/caffe-fast-rcnn/python"
    root_dir = "/media/foolyc/ssd/workspace/PycharmProjects/DQN-Atari-pycaffe-master/"
    solver_dir = "dqn_solver.prototxt"
    deploy_dir = ""
    model_dir =""
    sys.path.insert(0, pycaffe_dir)
    pass


def get_config():
    myconfig = DQNConfig
    return myconfig
