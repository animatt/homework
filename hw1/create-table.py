import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import sys


experiments, experts = [], []
if len(sys.argv) == 1:
    sys.exit('No arguments passed.')

for task in sys.argv[1:]:
    try:
        with open(f'./clones/{task}.results.pkl', 'rb') as f:
            experiments.append(pickle.load(f)[0])
    except FileNotFoundError:
        sys.exit(f"Couldn't find experimental data for {task}.")

    try:
        with open(f'./rollouts/{task}.results.pkl', 'rb') as f:
            experts.append(pickle.load(f))
    except FileNotFoundError:
        sys.exit(f"Couldn't find expert data for {task}.")


rows = ['expert performance', 'clone performance']
cols = sys.argv[1:]

txt = [[
    f'{expert["rw_mean"]:.4f} / {expert["rw_var"]:.4f}',
    f'{clone["rw_mean"]:.4f} / {clone["rw_var"]:.4f}'
] for clone, expert in zip(experiments, experts)]

txt = list(zip(*txt))

ax = plt.subplot(111)
ax.table(cellText=txt, rowLabels=rows, colLabels=cols, loc='center right')

table = ax.tables[0]
table.scale(xscale=1, yscale=2)
table.auto_set_font_size(value=False)
table.set_fontsize(size=10)

ax.axis('off')

plt.show()
