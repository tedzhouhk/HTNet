import os
import pickle
import random
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--setup', type=int, default=1, help='which setup to simulate')
parser.add_argument('--num_snapshot', type=int, default=10, help='number of snapshots in each sequence')
args = parser.parse_args()

if args.setup == 6:
    args.num_snapshot = 100

valid_fn = dict()
for fn in os.listdir('output/main/'):
    if fn.startswith('setup{}_'.format(args.setup)) and fn.endswith('.pkl'):
        valid_fn[fn.split('_')[1].split('.')[0]] = pickle.load(open('output/main/{}'.format(fn), 'rb'))
valid_fn['true'] = 1

fig, axs = plt.subplots(len(valid_fn), 1, figsize =(8, 15), tight_layout=True)
for d, ax in zip(list(valid_fn), axs):
    data = list()
    if d == 'true':
        for k in valid_fn['gbrt'].keys():
            data.append(valid_fn['gbrt'][k]['true'])
    else:
        for k in valid_fn[d].keys():
            data.append(valid_fn[d][k]['pred'])
    data = np.concatenate(data)
    ax.hist(data, bins=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40])
    ax.set_title(d)

plt.savefig('plot/plot.png',dpi=300)