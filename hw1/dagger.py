import sys
import gym
import pickle
import torch
import numpy as np
import tensorflow as tf
import tf_util
import load_policy
import itertools
from my_model import MyModel
from behavioral_cloning import load, train

if len(sys.argv) < 2:
    sys.exit('Missing task argument.')

task = sys.argv[1]

observations, actions = load(task)

# print('observations[:2] =', observations[:2])
# print('actions[:2] =', actions[:2])

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

with tf.Session():

    tf_util.initialize()

    for i_trial in range(NUM_TRIALS):
        model = MyModel(**params)
    
        train(model, observations, actions)

        print(f'observations.shape = {observations.shape}')
        print(f'actions.shape = {actions.shape}')

        rw_per_ep = []
        new_observations = []
    
        for ep_i in range(NUM_EPS):  # generate rollout
            obs = env.reset()
            done = False
            totalr = 0
    
            while not done:
    
                with torch.no_grad():
                    action = model(torch.from_numpy(obs).float())
                    obs, reward, done, info = env.step(action)
            
                new_observations.append(obs)
                totalr += reward
    
            rw_per_ep.append(totalr)

        rewards.append(np.mean(rw_per_ep))

        expert_data = [
            (obs, policy_fn(obs[None, :])) for obs in new_observations
        ]
        new_observations, new_actions = list(zip(*expert_data))

        new_observations = torch.tensor(
            [obs.tolist() for obs in new_observations]
        )
        new_actions = torch.tensor(list(itertools.chain.from_iterable(
            acts.tolist() for acts in new_actions
        )))

        observations = torch.cat((observations, new_observations), 0)
        actions = torch.cat((actions, new_actions), 0)

        print(f'Completed pass #{i_trial}')

print(rewards)
