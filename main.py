#!/usr/bin/env python
# --------------------------------------------------------
# DQN in pycaffe
# Copyright (c) 2016 SLXrobot
# Written by Chao Yu
# --------------------------------------------------------
import argparse
import config
from pydqn.agent import Agent
from pydqn import environment


def playatari():
    env = environment.SimpleGymEnvironment(myconfig)
    agent = Agent(myconfig, env)
    if args.is_train:
        agent.train()
    else:
        agent.play()
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('is_train', default=True, help="is train or not")
    # parser.add_argument('--display', default=True, help="display or not")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    myconfig = config.get_config()
    playatari()


