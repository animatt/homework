import sys
import torch
import gym
import pickle
import numpy as np
import pandas as pd
from utils import get_mean_var
from behavioral_cloning import load, train

from my_model import MyModel


def forward_prop(model, env, num_rollouts=1):
    returns = []
    for i_ep in range(num_rollouts):
        observation = env.reset()
    
        done = False
        totalr = 0
    
        while not done:
    
            # env.render()
            with torch.no_grad():
                action = model(torch.from_numpy(observation).float())
                observation, reward, done, info = env.step(action)
    
            totalr += reward
    
        returns.append(totalr)

    return {'rw_mean': np.mean(returns), 'rw_var': np.var(returns)}


if __name__ == '__main__':
    task = sys.argv[1]  # ie: Ant-v2
    
    env = gym.make(task)
    num_rollouts = 5
    
    try:
        with open(f'./clones/{task}.params.pkl', 'rb') as f:
            params = pickle.load(f)
    except FileNotFoundError:
        sys.exit(f"Couldn't find params for {task}.")
    
    model = MyModel(**params)
    model.load_state_dict(torch.load(f'./clones/{task}.pt'))
    
    experimental_results = [forward_prop(model, env, num_rollouts)]
    
    with open(f'./clones/{task}.results.pkl', 'wb') as f:
        pickle.dump(experimental_results, f)
    
