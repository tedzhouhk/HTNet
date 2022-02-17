import os
import torch
import dgl
import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm

num_snapshots = 10

def check_valid(prefix, i, fn):
    for snapshot in range(num_snapshots):
        if not os.path.isfile(prefix + '{}_{}_{}.csv'.format(i, snapshot, fn)):
            return False
        if not os.path.isfile(prefix + '{}_{}_{}.out'.format(i, snapshot, fn)):
            return False
        elif os.path.getsize(prefix + '{}_{}_{}.out'.format(i, snapshot, fn)) == 0:
            return False
    if not os.path.isfile(prefix + '{}_{}.pkl'.format(i, fn)):
        return False
    return True

def compute_dist(df, ap_map, sta_map):
    ap_x = torch.zeros(len(ap_map))
    ap_y = torch.zeros(len(ap_map))
    sta_x = torch.zeros(len(sta_map))
    sta_y = torch.zeros(len(sta_map))
    for r in df.rows:
        nc = r['node_code']
        if nc.startswith('A'):
            ap_x[ap_map[nc]] = float(r['x(m)'])
            ap_y[ap_map[nc]] = float(r['y(m)'])
        else:
            sta_x[ap_map[nc]] = float(r['x(m)'])
            sta_y[ap_map[nc]] = float(r['y(m)'])
    # compute ap-ap and ap-sta distance

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

if not os.path.exists('data/setup1/processed'):
    os.mkdir('data/setup1/processed')
for fn in tqdm(valid_fn):
    [i, raw_fn] = fn.split('!')
    # get AP/STA ids
    df = pd.read_csv('data/setup1/raw/{}_0_{}.csv'.format(i, raw_fn), sep=';')
    ap_map = dict()
    sta_map = dict()
    for nc in df['node_code'].tolist():
        if nc.startswith('A'):
            ap_map[nc] = len(ap_map)
        else:
            sta_map[nc] = len(sta_map)
    graphs = list()
    for j in range(num_snapshots):
        ap_feats = list()
        sta_feats = list()
        e_ap_feats = list()
        e_sta_feats = list()
        ap_mask = torch.zeros(len(ap_map), 1, dtype = torch.bool)
        sta_mask = torch.zeros(len(sta_map), 1, dtype = torch.bool)
        ap_throughput = torch.zeros(len(ap_map), 1)
        sta_throughput = torch.zeros(len(sta_throughput), 1)
        df = pd.read_csv('data/setup1/raw/{}_{}_{}.csv'.format(i, j, raw_fn), sep=';')
        ap_ap_dist, ap_sta_dist = compute_dist(pd, ap_map, sta_map)

    import pdb; pdb.set_trace()
