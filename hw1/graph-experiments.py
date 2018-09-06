import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import sys


experiments = []
if len(sys.argv) == 1:
    sys.exit('No arguments passed.')

for task in sys.argv[1:]:
    try:
        with open(f'./clones/{task}.results.pkl', 'rb') as f:
            experiments.append(pickle.load(f))
    except FileNotFoundError:
        sys.exit(f"Couldn't find data for {task}.")


rows = sys.argv[1:]   # [task for task in sys.argv[1:]]
cols = [trial for trial in experiments[0].keys()]
txt = [
    [
        f'{task[key]["rw_mean"]:.4f} /  {task[key]["rw_var"]:.4f}'
        for key in sorted(task)
    ] for task in experiments
]

print(txt)

ax = plt.subplot(111)
ax.table(cellText=txt, rowLabels=rows, colLabels=cols, loc='center right')

table = ax.tables[0]
table.scale(xscale=1, yscale=2)
table.auto_set_font_size(value=False)
table.set_fontsize(size=10)

ax.axis('off')

plt.show()
