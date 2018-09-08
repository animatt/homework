import sys
import gym
import pickle
import torch
import load_policy
from my_model import MyModel
from behavioral_cloning import load, train

if len(sys.argv) < 2:
    sys.exit('Missing task argument.')

task = sys.argv[1]

observations, actions = load(task)

try:
    with open(f'./clones/{task}.params.pkl', 'rb') as f:
        params = pickle.load(f)
except FileNotFoundError:
    sys.exit(f'Could not find {task} parameters.')

env = gym.make(task)
policy_fn = load_policy.load_policy(f'./experts/{task}.pkl')

NUM_EPS = 5
NUM_TRIALS = 5
rewards = []
for i_trial in range(NUM_TRIALS):
    model = MyModel(**params)
    model.load_state_dict(torch.load(f'./clones/{task}.pt'))

    train(model, observations, actions)

    rw_per_ep = []
    observations = []

    for ep_i in range(NUM_EPS):  # generate rollout
        obs = env.reset()
        done = False
        totalr = 0

        while not done:
            action = model(torch.from_numpy(obs).float())
            obs, reward, done, info = env.step(action)
        
            observations.append(obs)
            totalr += reward

        rw_per_ep.append(totalr)
        

    expert_data = [(obs, policy_fun(obs[None, :])) for obs in observations]
    expert_data = zip(*expert_data)

    sys.exit(expert_data)
