import os
import pickle
import random
import numpy as np
from collections import defaultdict

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--setup', type=int, help='which setup to simulate')
parser.add_argument('--num_snapshot', type=int, default=10, help='number of snapshots in each sequence')
args = parser.parse_args()

valid_fn = dict()
for fn in os.listdir('output/main/'):
    if fn.startswith('setup{}_'.format(args.setup)) and fn.endswith('.pkl'):
        valid_fn[fn.split('_')[1].split('.')[0]] = pickle.load(open('output/main/{}'.format(fn), 'rb'))

tot_throughput = 0
tot_number = 0
for key in valid_fn['sinr']:
    tot_throughput += valid_fn['sinr'][key]['true'].sum()
    tot_number += valid_fn['sinr'][key]['true'].shape[0]
print('Average throughput: {:.2f}'.format(tot_throughput / tot_number))

while True:
    data = defaultdict(list)
    first = True
    for prefix in list(valid_fn):
        outs = valid_fn[prefix]
        if first:
            idx = random.choice(list(outs))
            sta_idx = random.randrange(len(outs[idx]['true']) // args.num_snapshot)
            data['true'].append('true')
            data[prefix].append(prefix)
            for i in range(args.num_snapshot):
                curr_sta_idx = sta_idx + i * (len(outs[idx]['true']) // args.num_snapshot)
                data['true'].append(outs[idx]['true'][curr_sta_idx])
                data[prefix].append(outs[idx]['pred'][curr_sta_idx])
            first = False
        else:
            data[prefix].append(prefix)
            for i in range(args.num_snapshot):
                curr_sta_idx = sta_idx + i * (len(outs[idx]['true']) // args.num_snapshot)
                data[prefix].append(outs[idx]['pred'][curr_sta_idx])
    with open('extracted.out', 'w') as f:
        prefix_list = list(data)
        for i in range(len(data['true'])):
            line = list()
            for prefix in prefix_list:
                line.append(str(data[prefix][i]))
            line = '\t'.join(line)
            f.write(line + '\n')
    import pdb; pdb.set_trace()