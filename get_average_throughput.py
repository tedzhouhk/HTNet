import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='which dataset to use')
parser.add_argument('--num_snapshot', type=int, default=10, help='number of snapshot')
args = parser.parse_args()

import torch
from minibatch import get_dataloader
from itertools import chain

train_dataloader, valid_dataloader, test_dataloader = get_dataloader('data/{}/processed/'.format(args.data), 1, all_cuda=False)

tot_sta = 0
throughput_sta = torch.zeros((args.num_snapshot), dtype=torch.float32)

for g, _ in chain(train_dataloader, valid_dataloader, test_dataloader):
    mask = g.nodes['sta'].data['mask'] > 0
    true = g.nodes['sta'].data['throughput'][mask].reshape(args.num_snapshot, -1)
    tot_sta += true.shape[-1]
    throughput_sta += true.sum(dim=1)

avg_throughput = throughput_sta / tot_sta

print('Average throughput: {:.2f}'.format(avg_throughput.mean()))
print(avg_throughput)