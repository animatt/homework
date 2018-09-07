import sys
import pickle
import matplotlib.pyplot as plt


if len(sys.argv) < 2:
    sys.exit('Missing task argument')

task = sys.argv[1]

try:
    with open(f'./clones/{task}.results.pkl', 'rb') as f:
        results = pickle.load(f)
except FileNotFoundError:
    sys.exit(f"Couldn't find reward / var for {task}")

results_tup = [(x['rw_mean'], x['rw_var']) for x in results]
rw, var = list(zip(*results_tup))

n = len(rw)

plt.figure(1)

plt.subplot(211)
plt.plot(range(n), rw)

plt.subplot(212)
plt.plot(range(n), var)

plt.title(f'Reward and Variance of {task} with respect to learning rate')
plt.show()
