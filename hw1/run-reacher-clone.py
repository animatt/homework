import sys
import torch
import gym
import pickle
from my_model import MyModel


task = 'Reacher-v2'


try:
    with open(f'./clones/{task}.params.pkl', 'rb') as f:
        params = pickle.load(f)
except FileNotFoundError:
    sys.exit(f'File {task}.params.pkl does not exist')
else:
    D_in = params['D_in']
    D_out = params['D_out']
    H = params['H']


model = MyModel(D_in, H, D_out)
try:
    model.load_state_dict(torch.load(f'./clones/{task}.pt'))
except FileNotFoundError:
    sys.exit(f'File {task}.pkl does not exist')
else:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


env = gym.make(task)
