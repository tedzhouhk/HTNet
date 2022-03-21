import os
import math
import random
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--setup', type=int, help='which setup to simulate')
args = parser.parse_args()

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

def check_ap_range(x, y, ap_pos):
    min_ap_c = -1
    min_ap_r = float('inf')
    for c in ap_pos:
        r = ((x - ap_pos[c]['x']) ** 2 + (y - ap_pos[c]['y']) ** 2) ** 0.5
        if r < min_ap_r:
            min_ap_r = r
            min_ap_c = c
    return min_ap_c

if args.setup == 1:
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

if args.setup == 2:
    # setup 2: moving across different APs
    print('generating setup 2...')
    frac_moving = 0.5
    max_speed = 1 #m/snapshot
    min_speed = 0.1 #m/snapshot

    if not os.path.exists('data/setup2'):
        os.mkdir('data/setup2')
    if not os.path.exists('data/setup2/raw'):
        os.mkdir('data/setup2/raw')
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
                if k == 0:
                    df['moved'] = 0
                for j, idx in enumerate(moving_idxs):
                    c = df.loc[idx]['wlan_code']
                    x = df.loc[idx]['x(m)'] + speeds[j] * math.cos(rads[j])
                    y = df.loc[idx]['y(m)'] + speeds[j] * math.sin(rads[j])
                    r = ((x - ap_pos[c]['x']) ** 2 + (y - ap_pos[c]['y']) ** 2) ** 0.5
                    if r < ap_pos[c]['r']:
                        # move within AP
                        df.at[idx, 'x(m)'] = x
                        df.at[idx, 'y(m)'] = y
                        moved += 1
                    else:
                        new_ap_c = check_ap_range(x, y, ap_pos)
                        if new_ap_c != -1:
                            # move across AP
                            df.at[idx, 'x(m)'] = x
                            df.at[idx, 'y(m)'] = y
                            df.at[idx, 'wlan_code'] = new_ap_c
                            moved += 1
                        else:
                            # exceed AP ranges, not moving
                            unmoved += 1
                    if k ==0:
                        df.at[idx, 'moved'] = 1
                fn = 'data/setup2/raw/{}_{}_{}'.format(i, k, input_files[p].split('/')[-1])
                sorted_df = df.copy(deep=True)
                sorted_df.sort_values(by=['wlan_code', 'node_type'], inplace=True, ignore_index=True)
                if k == 0:
                    first_snapshot_moving_idxs = list(sorted_df[sorted_df['moved'] == 1].index)
                sorted_df.drop('moved', axis=1, inplace=True)
                sorted_df.to_csv(fn, index=False, sep=';')
                adf = sorted_df
                bdf = pd.read_csv(fn, sep=';')
            mfn = 'data/setup2/raw/{}_{}.pkl'.format(i, input_files[p].split('/')[-1][:-4])
            with open(mfn, 'wb') as f:
                pickle.dump(first_snapshot_moving_idxs, f)
    print('moved: {}%'.format(moved / (moved + unmoved) * 100))

if args.setup == 3:
    # setup 3: moving interference sources
    print('generating setup 3...')
    num_interference = 3
    max_speed = 3 #m/snapshot
    min_speed = 1 #m/snapshot

    if not os.path.exists('data/setup3'):
        os.mkdir('data/setup3')
    if not os.path.exists('data/setup3/raw'):
        os.mkdir('data/setup3/raw')
    for p in tqdm(range(len(input_files))):
        odf = pd.read_csv(input_files[p],sep=';')
        wlan_codes = list(set(odf['wlan_code'].tolist()))
        ap_pos = get_ap_pos(odf)
        for i in range(num_scenarios):
            df = odf.copy(deep=True).astype({'z(m)': 'float64'})
            target_idxs = list(df[df['node_type'] == 1].index)
            curr_wlan_code = chr(ord(df['wlan_code'].max()) + 1)
            fake_df = df.head(n=2 * num_interference).copy(deep=True)
            fake_rads = list()
            fake_speed = list()
            for ii in range(num_interference):
                ap_idx = 2 * ii
                sta_idx = 2 * ii + 1
                fake_df.at[ap_idx, 'node_code'] = 'FAKE_AP_{}'.format(curr_wlan_code)
                fake_df.at[sta_idx, 'node_code'] = 'FAKE_STA_{}1'.format(curr_wlan_code)
                fake_df.at[ap_idx, 'node_type'] = 0
                fake_df.at[sta_idx, 'node_type'] = 1
                fake_df.at[ap_idx, 'wlan_code'] = curr_wlan_code
                fake_df.at[sta_idx, 'wlan_code'] = curr_wlan_code
                x = random.uniform(df['x(m)'].min(), df['x(m)'].max())
                y = random.uniform(df['y(m)'].min(), df['y(m)'].max())
                fake_df.at[ap_idx, 'x(m)'] = x
                fake_df.at[sta_idx, 'x(m)'] = x
                fake_df.at[ap_idx, 'y(m)'] = y
                fake_df.at[sta_idx, 'y(m)'] = y
                fake_df.at[sta_idx, 'z(m)'] = 0.01
                fake_df.at[ap_idx, 'primary_channel'] = 0
                fake_df.at[sta_idx, 'primary_channel'] = 0
                fake_df.at[ap_idx, 'min_channel_allowed'] = 0
                fake_df.at[ap_idx, 'max_channel_allowed'] = 7
                fake_df.at[sta_idx, 'min_channel_allowed'] = 0
                fake_df.at[sta_idx, 'max_channel_allowed'] = 7
                fake_rads.append(random.uniform(0, 2 * math.pi))
                fake_rads.append(fake_rads[-1])
                fake_speed.append(random.uniform(min_speed, max_speed))
                fake_speed.append(fake_speed[-1])
                curr_wlan_code = chr(ord(curr_wlan_code) + 1)
            for k in range(num_snapshots):
                concat_df = pd.concat([df, fake_df])
                fn = 'data/setup3/raw/{}_{}_{}'.format(i, k, input_files[p].split('/')[-1])
                concat_df.to_csv(fn, index=False, sep=';')
                if k < num_snapshots - 1:
                    for fi in range(fake_df.shape[0]):
                        fake_df.at[fi, 'x(m)'] = fake_df.at[fi, 'x(m)'] + fake_speed[fi] * math.cos(fake_rads[fi])
                        fake_df.at[fi, 'y(m)'] = fake_df.at[fi, 'y(m)'] + fake_speed[fi] * math.sin(fake_rads[fi])
            mfn = 'data/setup3/raw/{}_{}.pkl'.format(i, input_files[p].split('/')[-1][:-4])
            with open(mfn, 'wb') as f:
                pickle.dump(target_idxs, f)

def allocate_dynamic_channel(df):
    for idx, row in df.iterrows():
        if row['node_type'] == 0:
            p_channel = row['primary_channel']
            min_channel = row['min_channel_allowed']
            max_channel = row['max_channel_allowed']
            action = random.randrange(5)
            if action == 0:
                # increase min_channel_allowed
                if min_channel != max_channel:
                    min_channel += 1
                    if p_channel < min_channel:
                        p_channel = min_channel
            elif action == 1:
                # decrease min_channel_allowed
                if min_channel != 0:
                    min_channel -= 1
            elif action == 2:
                # increase max_channel_allowed:
                if max_channel != 7:
                    max_channel += 1
            elif action == 3:
                # decrease max_channel_allowed:
                if max_channel != min_channel:
                    max_channel -= 1
                    if p_channel > max_channel:
                        p_channel = max_channel
            elif action == 4:
                # random channels
                min_channel = random.randrange(8)
                max_channel = min_channel + random.randrange(8 - min_channel)
                p_channel = min_channel
            assert(min_channel >= 0 and min_channel <=7)
            assert(max_channel >= 0 and max_channel <=7)
            assert(min_channel <= max_channel)
            assert(min_channel <= p_channel)
            assert(p_channel <= max_channel)
        df.at[idx, 'primary_channel'] = p_channel
        df.at[idx, 'min_channel_allowed'] = min_channel
        df.at[idx, 'max_channel_allowed'] = max_channel

if args.setup == 4:
    # setup 4: dynamic channel allocation
    print('generating setup 4...')
    if not os.path.exists('data/setup4'):
        os.mkdir('data/setup4')
    if not os.path.exists('data/setup4/raw'):
        os.mkdir('data/setup4/raw')
    for p in tqdm(range(len(input_files))):
        odf = pd.read_csv(input_files[p],sep=';')
        wlan_codes = list(set(odf['wlan_code'].tolist()))
        ap_pos = get_ap_pos(odf)
        for i in range(num_scenarios):
            df = odf.copy(deep=True).astype({'z(m)': 'float64'})
            target_idxs = list(df[df['node_type'] == 1].index)
            for k in range(num_snapshots):
                allocate_dynamic_channel(df)                            
                fn = 'data/setup4/raw/{}_{}_{}'.format(i, k, input_files[p].split('/')[-1])
                df.to_csv(fn, index=False, sep=';')
            mfn = 'data/setup4/raw/{}_{}.pkl'.format(i, input_files[p].split('/')[-1][:-4])
            with open(mfn, 'wb') as f:
                pickle.dump(target_idxs, f)

if args.setup == 5:
    # setup 5: cross-AP moving sta + dynamic channel allocation (2 + 4)
    print('generating setup 5...')
    frac_moving = 0.5
    max_speed = 1 #m/snapshot
    min_speed = 0.1 #m/snapshot

    if not os.path.exists('data/setup5'):
        os.mkdir('data/setup5')
    if not os.path.exists('data/setup5/raw'):
        os.mkdir('data/setup5/raw')
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
                if k == 0:
                    df['moved'] = 0
                for j, idx in enumerate(moving_idxs):
                    c = df.loc[idx]['wlan_code']
                    x = df.loc[idx]['x(m)'] + speeds[j] * math.cos(rads[j])
                    y = df.loc[idx]['y(m)'] + speeds[j] * math.sin(rads[j])
                    r = ((x - ap_pos[c]['x']) ** 2 + (y - ap_pos[c]['y']) ** 2) ** 0.5
                    if r < ap_pos[c]['r']:
                        # move within AP
                        df.at[idx, 'x(m)'] = x
                        df.at[idx, 'y(m)'] = y
                        moved += 1
                    else:
                        new_ap_c = check_ap_range(x, y, ap_pos)
                        if new_ap_c != -1:
                            # move across AP
                            df.at[idx, 'x(m)'] = x
                            df.at[idx, 'y(m)'] = y
                            df.at[idx, 'wlan_code'] = new_ap_c
                            moved += 1
                        else:
                            # exceed AP ranges, not moving
                            unmoved += 1
                    if k ==0:
                        df.at[idx, 'moved'] = 1
                fn = 'data/setup5/raw/{}_{}_{}'.format(i, k, input_files[p].split('/')[-1])
                sorted_df = df.copy(deep=True)
                sorted_df.sort_values(by=['wlan_code', 'node_type'], inplace=True, ignore_index=True)
                if k == 0:
                    first_snapshot_moving_idxs = list(sorted_df[sorted_df['moved'] == 1].index)
                sorted_df.drop('moved', axis=1, inplace=True)
                allocate_dynamic_channel(sorted_df)
                sorted_df.to_csv(fn, index=False, sep=';')
                adf = sorted_df
                bdf = pd.read_csv(fn, sep=';')
            mfn = 'data/setup5/raw/{}_{}.pkl'.format(i, input_files[p].split('/')[-1][:-4])
            with open(mfn, 'wb') as f:
                pickle.dump(first_snapshot_moving_idxs, f)
    print('moved: {}%'.format(moved / (moved + unmoved) * 100))

if args.setup == 6:
    # setup 5: setup 5 + long sequence (100)
    num_snapshots = 100
    num_scenarios = 1

    print('generating setup 6...')
    frac_moving = 0.5
    max_speed = 1 #m/snapshot
    min_speed = 0.1 #m/snapshot

    if not os.path.exists('data/setup6'):
        os.mkdir('data/setup6')
    if not os.path.exists('data/setup6/raw'):
        os.mkdir('data/setup6/raw')
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
                if k == 0:
                    df['moved'] = 0
                for j, idx in enumerate(moving_idxs):
                    c = df.loc[idx]['wlan_code']
                    x = df.loc[idx]['x(m)'] + speeds[j] * math.cos(rads[j])
                    y = df.loc[idx]['y(m)'] + speeds[j] * math.sin(rads[j])
                    r = ((x - ap_pos[c]['x']) ** 2 + (y - ap_pos[c]['y']) ** 2) ** 0.5
                    if r < ap_pos[c]['r']:
                        # move within AP
                        df.at[idx, 'x(m)'] = x
                        df.at[idx, 'y(m)'] = y
                        moved += 1
                    else:
                        new_ap_c = check_ap_range(x, y, ap_pos)
                        if new_ap_c != -1:
                            # move across AP
                            df.at[idx, 'x(m)'] = x
                            df.at[idx, 'y(m)'] = y
                            df.at[idx, 'wlan_code'] = new_ap_c
                            moved += 1
                        else:
                            # exceed AP ranges, not moving
                            unmoved += 1
                    if k ==0:
                        df.at[idx, 'moved'] = 1
                fn = 'data/setup6/raw/{}_{}_{}'.format(i, k, input_files[p].split('/')[-1])
                sorted_df = df.copy(deep=True)
                sorted_df.sort_values(by=['wlan_code', 'node_type'], inplace=True, ignore_index=True)
                if k == 0:
                    first_snapshot_moving_idxs = list(sorted_df[sorted_df['moved'] == 1].index)
                sorted_df.drop('moved', axis=1, inplace=True)
                allocate_dynamic_channel(sorted_df)
                sorted_df.to_csv(fn, index=False, sep=';')
                adf = sorted_df
                bdf = pd.read_csv(fn, sep=';')
            mfn = 'data/setup6/raw/{}_{}.pkl'.format(i, input_files[p].split('/')[-1][:-4])
            with open(mfn, 'wb') as f:
                pickle.dump(first_snapshot_moving_idxs, f)
    print('moved: {}%'.format(moved / (moved + unmoved) * 100))