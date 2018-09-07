import sys
import gym
import torch
import pickle
from my_model import MyModel
from run_clone import forward_prop
from behavioral_cloning import load, train


if len(sys.argv) < 2:
    sys.exit('No arguments supplied.')
else:
    task = sys.argv[1]

try:
    observations, actions = load(task)
except FileNotFoundError:
    sys.exit(f'No rollouts for {task}')

with open(f'./clones/{task}.params.pkl', 'rb') as f:
    model_params = pickle.load(f)

env = gym.make(task)

experimental_results = []
passes = 5
lrs = [(x + 1) / 100 for x in range(passes)]
for i_pass, lr in enumerate(lrs, start=1):
    model = MyModel(**model_params)

    train(model, observations, actions, learning_rate=lr, verbose=False)
    result = forward_prop(model, env, num_rollouts=5)

    experimental_results.append(result)
    print(f'Completed pass #{i_pass}')


with open(f'./clones/{task}.results.pkl', 'wb') as f:
    pickle.dump(experimental_results, f)
