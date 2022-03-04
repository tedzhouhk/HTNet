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
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--output', type=str, default='', help='save predictions to files')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import dgl
import pickle
import numpy as np
from minibatch import get_dataloader
from model import NetModel
from datetime import datetime

train_dataloader, valid_dataloader, test_dataloader = get_dataloader('data/{}/processed/'.format(args.data), args.batch_size, all_cuda=True)

model = NetModel(args.layer, args.dim, args.graph, args.hetero, args.dynamic, args.num_snapshot).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def eval(model, dataloader, output=''):
    if output != '':
        if not os.path.exists('output'):
            os.mkdir('output')
        output_fn = 'output/{}.pkl'.format(output)
        outs = dict()
    model.eval()
    rmse = list()
    rmse_tot = 0
    with torch.no_grad():
        for g, idx in dataloader:
            g = g.to('cuda')
            mask = g.nodes['sta'].data['mask'] > 0
            pred = model(g)[mask]
            true = g.nodes['sta'].data['throughput'][mask]
            loss = torch.sqrt(torch.nn.functional.mse_loss(pred, true) + 1e-8)
            rmse.append(float(loss) * pred.shape[0])
            rmse_tot += pred.shape[0]
            if output != '':
                out = {'pred':pred.cpu().detach().numpy(), 'true':treu.cpu().detach().numpy()}
                outs[idx] = out
    if output != '':
        with open(output_fn, 'wb') as f:
            pickle.dump(outs, f)
    return np.sum(np.array(rmse)) / rmse_tot

# torch.autograd.set_detect_anomaly(True)
best_e = 0
best_valid_rmse = float('inf')
model_fn = 'models/{}.pkl'.format(datetime.now().strftime('%m-%d-%H:%M:%S'))
if not os.path.exists('models'):
    os.mkdir('models')
for e in range(args.epoch):
    model.train()
    train_rmse = list()
    rmse_tot = 0
    for g, _ in train_dataloader:
        g = g.to('cuda')
        optimizer.zero_grad()
        mask = g.nodes['sta'].data['mask'] > 0
        pred = model(g)[mask]
        true = g.nodes['sta'].data['throughput'][mask]
        loss = torch.sqrt(torch.nn.functional.mse_loss(pred, true) + 1e-8)
        train_rmse.append(float(loss) * pred.shape[0])
        rmse_tot += pred.shape[0]
        # with torch.autograd.detect_anomaly():
        loss.backward()
        optimizer.step()
    train_rmse = np.sum(np.array(train_rmse)) / rmse_tot
    valid_rmse = eval(model, valid_dataloader)
    print('Epoch: {} Training RMSE: {:.4f} Validation RMSE: {:.4f}'.format(e, train_rmse, valid_rmse))
    if valid_rmse < best_valid_rmse:
        best_e = e
        best_valid_rmse = valid_rmse
        torch.save(model.state_dict(), model_fn)

print('Loading model in epoch {}...'.format(best_e))
model.load_state_dict(torch.load(model_fn))
print('Test RMSE: {:.4f}'.format(eval(model, test_dataloader, output=args.output)))