import os
import dgl
import pandas as pd
import numpy as np
from tqdm import tqdm

num_snapshots = 10

def check_valid(prefix, i, fn):
    for snapshot in range(num_snapshots):
        if not os.path.isfile(prefix + '{}_{}_{}.csv'.format(i, snapshot, fn)):
            return False
        if not os.path.isfile(prefix + '{}_{}_{}.out'.format(i, snapshot, fn)):  
            return False
    if not os.path.isfile(prefix + '{}_{}.pkl'.format(i, fn)):
        return False
    return True

# setup 1
valid_fn = dict()
for fn in os.listdir('data/setup1/raw'):
    if fn.endswith('csv'):
        i = fn.split('_')[0]
        raw_fn = fn.split('_', 2)[2][:-4]
        if not '{}!{}'.format(i, raw_fn) in valid_fn:
            if check_valid('data/setup1/raw/', i, raw_fn):
                valid_fn['{}!{}'.format(i, raw_fn)] = len(valid_fn)
print('Total training samples:', len(valid_fn))