import sys
import torch
import gym
import pickle
import pandas as pd
from behavioral_cloning import load, train

from my_model import MyModel


task = sys.argv[1]  # ie: Ant-v2

observations, actions = load(task)

N, D_in = observations.shape
_, D_out = actions.shape
H = 100


env = gym.make(task)
num_trials = 2
per_rollout = env.spec.timestep_limit


for i_trial in range(1, num_trials + 1):
    model = MyModel(D_in, H, D_out)

    current_size = i_trial * per_rollout
    train(model, observations[:current_size], actions[:current_size])

    for i_ep in range(20):
        observation = env.reset()
        for t in range(per_rollout):

            env.render()
            with torch.no_grad():
                action = model(torch.from_numpy(observation).float())
                observation, reward, done, info = env.step(action)

            if t % 100 == 0:
                print(f'reward = {reward}')
            if done:
                print(f'Episode finished after {t} timesteps')
                break
