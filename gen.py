import os
import math
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

num_snapshots = 10
num_scenarios = 10

input_files = list()
for sce in ['sce1a','sce1b','sce1c','sce2b','sce2c']:
    for f in os.listdir('ITU/' + sce):
        input_files.append('ITU/' + sce + '/' + f)

if not os.path.exists('data'):
    os.mkdir('data')

def get_ap_pos(df):
    wlan_codes = list(set(df['wlan_code'].tolist()))
    ap_pos = dict()
    for c in wlan_codes:
        xap = float(df[(df['wlan_code'] == c) & (df['node_type'] == 0)]['x(m)'])
        yap = float(df[(df['wlan_code'] == c) & (df['node_type'] == 0)]['y(m)'])
        rap = 0
        for _, row in df[(df['wlan_code'] == c) & (df['node_type'] == 1)].iterrows():
            xsta = row['x(m)']
            ysta = row['y(m)']
            if ((xsta - xap) ** 2 + (ysta - yap) ** 2) ** 0.5 > rap:
                rap = ((xsta - xap) ** 2 + (ysta - yap) ** 2) ** 0.5
        ap_pos[c] = {'x':xap, 'y':yap, 'r':rap}
    return ap_pos


# setup 1: moving within the covering range of AP
print('generating setup 1...')
frac_moving = 0.5
max_speed = 0.5 #m/snapshot
min_speed = 0.1 #m/snapshot

if not os.path.exists('data/setup1'):
    os.mkdir('data/setup1')
if not os.path.exists('data/setup1/raw'):
    os.mkdir('data/setup1/raw')
moved = 0
unmoved = 0
for p in tqdm(range(len(input_files))):
    odf = pd.read_csv(input_files[p],sep=';')
    wlan_codes = list(set(odf['wlan_code'].tolist()))
    ap_pos = get_ap_pos(odf)
    for i in range(num_scenarios):
        df = odf.copy(deep=True)
        moving_idxs = list()
        for c in wlan_codes:
            moving_idxs += list(df[(df['wlan_code'] == c) & (df['node_type'] == 1)].sample(frac=frac_moving).index.values)
        speeds = np.random.uniform(min_speed, max_speed, len(moving_idxs))
        rads = np.random.uniform(0, 2 * math.pi, len(moving_idxs))
        for k in range(num_snapshots):
            for j, idx in enumerate(moving_idxs):
                c = df.loc[idx]['wlan_code']
                x = df.loc[idx]['x(m)'] + speeds[j] * math.cos(rads[j])
                y = df.loc[idx]['y(m)'] + speeds[j] * math.sin(rads[j])
                r = ((x - ap_pos[c]['x']) ** 2 + (y - ap_pos[c]['y']) ** 2) ** 0.5
                if r < ap_pos[c]['r']:
                    df.at[idx, 'x(m)'] = x
                    df.at[idx, 'y(m)'] = y
                    moved += 1
                else:
                    unmoved += 1
            fn = 'data/setup1/raw/{}_{}_{}'.format(i, k, input_files[p].split('/')[-1])
            df.to_csv(fn, index=False, sep=';')
        mfn = 'data/setup1/raw/{}_{}.pkl'.format(i, input_files[p].split('/')[-1][:-4])
        with open(mfn, 'wb') as f:
            pickle.dump(moving_idxs, f)
print('moved: {}%'.format(moved / (moved + unmoved) * 100))