'''
Implementation of behavioral cloning for Berkeley CS294-112
The first argument to behavioral-cloning.py should be the task,
for instance: $ python behavioral-cloning.py Ant-v2
'''

import torch
import pickle
import sys

from my_model import MyModel


def load(task):
    expert_path = f'./rollouts/{task}.pkl'
    try:
        with open(expert_path, 'rb') as f:
            expert_pol = pickle.load(f)
    except FileNotFoundError:
        sys.exit(f'File {expert_path} does not exist')
    else:
        obs = torch.from_numpy(expert_pol['observations']).float()
        act = torch.from_numpy(expert_pol['actions']).float()[:, 0]

        return (obs, act)


def train(model, observations, actions, learning_rate=1e-2):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(700):
        actions_pred = model(observations)

        loss = loss_fn(actions_pred, actions)
        print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    task = sys.argv[1]
    observations, actions = load(task)


    N, D_in = observations.shape
    _, D_out = actions.shape
    H1 = 100


    model = MyModel(D_in, H1, D_out)
    train(model, observations, actions, learning_rate=1e-2)


    torch.save(model.state_dict(), f'./clones/{task}.pt')
    with open(f'./clones/{task}.params.pkl', 'wb') as f:
        pickle.dump({'D_in': D_in, 'D_out': D_out, 'H': H1}, f)
