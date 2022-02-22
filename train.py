import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--data', type=str, help='which dataset to use')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_snapshot', type=int, default=10, help='number of snapshot')
parser.add_argument('--graph', action='store_true', help='whether to use graph information')
parser.add_argument('--hetero', action='store_true', help='whether to treat as heterogeneous graph')
parser.add_argument('--dynamic', action='store_true', help='whether to use dynamic information')
parser.add_argument('--layer', type=int, default=2, help='number of layers')
parser.add_argument('--dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import dgl
import numpy as np
from minibatch import get_dataloader
from model import NetModel
from datetime import datetime

train_dataloader, valid_dataloader, test_dataloader = get_dataloader('data/{}/processed/'.format(args.data), args.batch_size, all_cuda=True)

model = NetModel(args.layer, args.dim, args.graph, args.hetero, args.dynamic, args.num_snapshot).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def eval(model, dataloader):
    model.eval()
    rmse = list()
    with torch.no_grad():
        for g in dataloader:
            g = g.to('cuda')
            mask = g.nodes['sta'].data['mask'] > 0
            pred = model(g)[mask]
            true = g.nodes['sta'].data['throughput'][mask]
            loss = torch.sqrt(torch.nn.functional.mse_loss(pred, true) + 1e-8)
            rmse.append(float(loss))
    return np.mean(np.array(rmse))

# torch.autograd.set_detect_anomaly(True)
best_e = 0
best_valid_rmse = float('inf')
model_fn = 'models/{}.pkl'.format(datetime.now().strftime('%m-%d-%H:%M:%S'))
if not os.path.exists('models'):
    os.mkdir('models')
for e in range(args.epoch):
    model.train()
    train_rmse = list()
    for g in train_dataloader:
        g = g.to('cuda')
        optimizer.zero_grad()
        mask = g.nodes['sta'].data['mask'] > 0
        pred = model(g)[mask]
        true = g.nodes['sta'].data['throughput'][mask]
        loss = torch.sqrt(torch.nn.functional.mse_loss(pred, true) + 1e-8)
        train_rmse.append(float(loss))
        # with torch.autograd.detect_anomaly():
        loss.backward()
        optimizer.step()
    train_rmse = np.mean(np.array(train_rmse))
    valid_rmse = eval(model, valid_dataloader)
    print('Epoch: {:.2f} Training RMSE: {:.2f} Validation RMSE: {:.2f}'.format(e, train_rmse, valid_rmse))
    if valid_rmse < best_valid_rmse:
        best_e = e
        best_valid_rmse = valid_rmse
        torch.save(model.state_dict(), model_fn)

print('Loading model in epoch {}...'.format(best_e))
model.load_state_dict(torch.load(model_fn))
print('Test RMSE: {:.2f}'.format(eval(model, test_dataloader)))
