import os
import torch
import dgl
import pickle
import random
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

def get_dist(df, ap_map, sta_map):
    ap_x = torch.zeros(len(ap_map))
    ap_y = torch.zeros(len(ap_map))
    sta_x = torch.zeros(len(sta_map))
    sta_y = torch.zeros(len(sta_map))
    for _, r in df.iterrows():
        nc = r['node_code']
        if nc.startswith('A'):
            ap_x[ap_map[nc]] = float(r['x(m)'])
            ap_y[ap_map[nc]] = float(r['y(m)'])
        else:
            sta_x[sta_map[nc]] = float(r['x(m)'])
            sta_y[sta_map[nc]] = float(r['y(m)'])
    # compute ap-ap and ap-sta distance
    ap_loc = torch.stack((ap_x, ap_y), dim=1)
    sta_loc = torch.stack((sta_x, sta_y), dim=1)
    return ap_x, ap_y, sta_x, sta_y, torch.cdist(ap_loc, ap_loc), torch.cdist(ap_loc, sta_loc)

def get_channel(df, ap_map, sta_map):
    ap_pchannel = torch.zeros((len(ap_map), 8))
    ap_channel = torch.zeros((len(ap_map), 8))
    sta_pchannel = torch.zeros((len(sta_map), 8))
    sta_channel = torch.zeros((len(sta_map), 8))
    for _, r in df.iterrows():
        nc = r['node_code']
        pc = r['primary_channel']
        minc = r['min_channel_allowed']
        maxc = r['max_channel_allowed']
        if nc.startswith('A'):
            ap_pchannel[ap_map[nc]][pc] = 1
            ap_channel[ap_map[nc]][minc:maxc + 1] = 1
        else:
            sta_pchannel[sta_map[nc]][pc] = 1
            sta_channel[sta_map[nc]][minc:maxc + 1] = 1
    return ap_pchannel, ap_channel, sta_pchannel, sta_channel

def get_simulated_result(df, ap_map, sta_map, out_fn):
    sta_throughput = torch.zeros(len(sta_map))
    ap_throughput = torch.zeros(len(ap_map))
    ap_airtime = torch.zeros(len(ap_map))
    sta_sinr = torch.zeros(len(sta_map))
    ap_sta_rssi = torch.zeros((len(ap_map), len(sta_map)))
    ap_ap_inter = torch.zeros((len(ap_map), len(ap_map)))
    with open(out_fn, 'r') as f:
        f.readline()
        throughput = f.readline().strip('{}\n').split(',')
        airtime_raw = f.readline().strip('{}\n')[:-1].split(';')
        airtime = list()
        for ar in airtime_raw:
            split_rar = ar.split(',')
            split_ar = [float(x) for x in split_rar]
            airtime.append(sum(split_ar) / len(split_ar))
        rssi = f.readline().strip('{}\n').split(',')
        inter = list()
        for _ in range(len(ap_map)):
            inter.append(f.readline().strip('{}\n;').split(','))
        sinr = f.readline().strip('{}\n').split(',')
    ap_neighbors_nc = df[df['node_type']==0]['node_code'].tolist()
    ap_neighbors = [ap_map[x] for x in ap_neighbors_nc]
    curr_ap = 0
    for i, r in df.iterrows():
        nc = r['node_code']
        if nc.startswith('A'):
            ap_throughput[ap_map[nc]] = float(throughput[i])
            ap_airtime[ap_map[nc]] = float(airtime[curr_ap])
            for j, ap in enumerate(ap_neighbors):
                ap_ap_inter[ap_map[nc]][ap] = float(inter[curr_ap][j])
            curr_ap += 1
        else:
            sta_throughput[sta_map[nc]] = float(throughput[i])
            sta_sinr[sta_map[nc]] = float(sinr[i])
            ap = ap_map[df[(df['node_type'] == 0) & (df['wlan_code'] == r['wlan_code'])]['node_code'].tolist()[0]]
            ap_sta_rssi[ap][sta_map[nc]] = float(rssi[i])
    return sta_throughput, ap_throughput, ap_airtime, sta_sinr, ap_sta_rssi, ap_ap_inter

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

tot_idx = list(range(len(valid_fn)))
random.shuffle(tot_idx)
train_idx = set(tot_idx[:int(len(valid_fn) * 0.6)])
valid_idx = set(tot_idx[int(len(valid_fn) * 0.6):int(len(valid_fn) * 0.8)])
test_idx = set(tot_idx[int(len(valid_fn) * 0.8):])
curr_train = 0
curr_valid = 0
curr_test = 0

for fn_idx, fn in tqdm(enumerate(valid_fn), total=len(valid_fn)):
    [ii, raw_fn] = fn.split('!')
    # get AP/STA ids
    df = pd.read_csv('data/setup1/raw/{}_0_{}.csv'.format(ii, raw_fn), sep=';')
    ap_map = dict()
    sta_map = dict()
    for nc in df['node_code'].tolist():
        if nc.startswith('A'):
            ap_map[nc] = len(ap_map)
        else:
            sta_map[nc] = len(sta_map)
    ap_mask = torch.zeros((len(ap_map), 1), dtype = torch.bool)
    sta_mask = torch.zeros((len(sta_map), 1), dtype = torch.bool)
    moving_idxs = pickle.load(open('data/setup1/raw/{}_{}.pkl'.format(ii, raw_fn), 'rb'))
    for idx in moving_idxs:
        sta_mask[sta_map[df.at[idx,'node_code']]] = 1
    graphs = list()
    for j in range(num_snapshots):
        df = pd.read_csv('data/setup1/raw/{}_{}_{}.csv'.format(ii, j, raw_fn), sep=';')
        ap_x, ap_y, sta_x, sta_y, ap_ap_dist, ap_sta_dist = get_dist(df, ap_map, sta_map)
        ap_pchannel, ap_channel, sta_pchannel, sta_channel = get_channel(df, ap_map, sta_map)
        sta_throughput, ap_throughput, ap_airtime, sta_sinr, ap_sta_rssi, ap_ap_inter = get_simulated_result(df, ap_map, sta_map, 'data/setup1/raw/{}_{}_{}.out'.format(ii, j, raw_fn))

        ap_throughput = ap_throughput.unsqueeze(-1)
        sta_throughput = sta_throughput.unsqueeze(-1)

        ap_feat = torch.zeros((len(ap_map), 21))
        for i in range(len(ap_map)):
            ap_feat[i][1] = ap_x[i] / 100
            ap_feat[i][2] = ap_y[i] / 100
            ap_feat[i][3:11] = ap_pchannel[i]
            ap_feat[i][11:19] = ap_channel[i]
            ap_feat[i][19] = ap_airtime[i] / 100
        
        sta_feat = torch.zeros((len(sta_map), 21))
        sta_feat[:,0] = 1
        for i in range(len(sta_map)):
            sta_feat[i][1] = sta_x[i] / 100
            sta_feat[i][2] = sta_y[i] / 100
            sta_feat[i][3:11] = sta_pchannel[i]
            sta_feat[i][11:19] = sta_channel[i]
            sta_feat[i][20] = sta_sinr[i] / 100

        ap_ap_edge_src = torch.zeros(len(ap_map) * (len(ap_map) - 1), dtype=torch.int64)
        ap_ap_edge_dst = torch.zeros(len(ap_map) * (len(ap_map) - 1), dtype=torch.int64)
        ap_ap_edge_feat = torch.zeros((ap_ap_edge_src.shape[0]), 4)
        ap_ap_edge_feat[:,0] = 1
        curr_e = 0
        for i in range(len(ap_map)):
            for j in range(len(ap_map)):
                if i != j:
                    ap_ap_edge_src[curr_e] = i
                    ap_ap_edge_dst[curr_e] = j
                    ap_ap_edge_feat[curr_e][1] = ap_ap_dist[i][j] / 100
                    ap_ap_edge_feat[curr_e][3] = ap_ap_inter[i][j] / 100
                    curr_e += 1

        ap_sta_edge_src = torch.zeros(len(sta_map), dtype=torch.int64)
        ap_sta_edge_dst = torch.zeros(len(sta_map), dtype=torch.int64)
        ap_sta_edge_feat = torch.zeros((len(sta_map), 4))
        sta_ap_edge_src = torch.zeros(len(sta_map), dtype=torch.int64)
        sta_ap_edge_dst = torch.zeros(len(sta_map), dtype=torch.int64)
        sta_ap_edge_feat = torch.zeros((len(sta_map), 4))
        curr_e = 0
        for _, r in df.iterrows():
            if r['node_type'] == 1:
                sta_idx = sta_map[r['node_code']]
                ap_idx = ap_map[df[(df['node_type'] == 0) & (df['wlan_code'] == r['wlan_code'])]['node_code'].tolist()[0]]
                ap_sta_edge_src[curr_e] = ap_idx
                ap_sta_edge_dst[curr_e] = sta_idx
                sta_ap_edge_src[curr_e] = sta_idx
                sta_ap_edge_dst[curr_e] = ap_idx
                ap_sta_edge_feat[curr_e][1] = ap_sta_dist[ap_idx][sta_idx] / 100
                ap_sta_edge_feat[curr_e][2] = ap_sta_rssi[ap_idx][sta_idx] / 100
                sta_ap_edge_feat[curr_e][1] = ap_sta_dist[ap_idx][sta_idx] / 100
                sta_ap_edge_feat[curr_e][2] = ap_sta_rssi[ap_idx][sta_idx] / 100
                curr_e += 1

        g = dgl.heterograph({
            ('ap', 'ap_ap', 'ap'): (ap_ap_edge_src, ap_ap_edge_dst),
            ('ap', 'ap_sta', 'sta'): (ap_sta_edge_src, ap_sta_edge_dst),    
            ('sta', 'sta_ap', 'ap'): (sta_ap_edge_src, sta_ap_edge_dst)    
        })
        g.nodes['ap'].data['feat'] = ap_feat
        g.nodes['ap'].data['mask'] = ap_mask
        g.nodes['ap'].data['throughput'] = ap_throughput
        g.nodes['sta'].data['feat'] = sta_feat
        g.nodes['sta'].data['mask'] = sta_mask
        g.nodes['sta'].data['throughput'] = sta_throughput
        g.edges['ap_ap'].data['feat'] = ap_ap_edge_feat
        g.edges['ap_sta'].data['feat'] = ap_sta_edge_feat
        g.edges['sta_ap'].data['feat'] = sta_ap_edge_feat
        graphs.append(g)
    graphs = [dgl.batch(graphs)]
    if fn_idx in train_idx:
        dgl.data.utils.save_graphs('data/setup1/processed/train_{}.bin'.format(curr_train), graphs, {"g": torch.tensor([0])})
        curr_train += 1
    elif fn_idx in valid_idx:
        dgl.data.utils.save_graphs('data/setup1/processed/valid_{}.bin'.format(curr_valid), graphs, {"g": torch.tensor([0])})
        curr_valid += 1
    else:
        dgl.data.utils.save_graphs('data/setup1/processed/test_{}.bin'.format(curr_test), graphs, {"g": torch.tensor([0])})
        curr_test += 1
