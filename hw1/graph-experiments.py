import pickle
import sys


experiments = []
for task in sys.argv[1:]:
    try:
        with open(f'./clones/{task}.results.pkl', 'rb') as f:
            experiments.append(pickle.load(f))
    except FileNotFoundError:
        sys.exit(f"Couldn't find data for {task}")

print(experiments)
