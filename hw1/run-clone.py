import sys
import torch
import gym
import pickle
import pandas as pd
from utils import get_mean_var
from behavioral_cloning import load, train

from my_model import MyModel


task = sys.argv[1]  # ie: Ant-v2

observations, actions = load(task)

N, D_in = observations.shape
_, D_out = actions.shape
num_eps = 4
H = 100


env = gym.make(task)
num_trials = 2
per_rollout = env.spec.timestep_limit
experimental_results = []


for i in range(num_trials):
    experimental_results.append([])


for i_trial in range(1, num_trials + 1):
    model = MyModel(D_in, H, D_out)

    rw_mean, rw_var = 0, 0
    current_size = i_trial * per_rollout
    train(model, observations[:current_size], actions[:current_size])

    for i_ep in range(num_eps):
        observation = env.reset()
        for t in range(per_rollout):

            # env.render()
            with torch.no_grad():
                action = model(torch.from_numpy(observation).float())
                observation, reward, done, info = env.step(action)

            rw_mean, rw_var = get_mean_var(reward, rw_mean, rw_var, t + 1)

            if t % 100 == 0:
                print(f'reward = {reward}')
                print(f'rw_mean = {rw_mean}\nrw_var = {rw_var}\n')
            if done:
                print(f'Episode finished after {t} timesteps')
                break

    experimental_results[i_trial - 1].append((rw_mean, rw_var))


with open(f'./clones/{task}.results.pkl', 'wb') as f:
    pickle.dump(experimental_results, f)
