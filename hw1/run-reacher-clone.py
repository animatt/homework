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
for i_ep in range(20):
    # observation = torch.from_numpy(env.reset()).float()
    observation = env.reset()
    for t in range(env.spec.timestep_limit):
        env.render()
        action = model(torch.from_numpy(observation).float())

        observation, reward, done, info = env.step(action)
        if done:
            print(f'Episode finished after {t} timesteps')
            break
