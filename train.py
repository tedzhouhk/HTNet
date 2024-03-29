import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--data', type=str, help='which dataset to use')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_snapshot', type=int, default=10, help='number of snapshot')
parser.add_argument('--sinr', action='store_true', help='whether to use linear regression based on SINR (this only works on CPU)')
parser.add_argument('--gbrt', action='store_true', help='whether to use gradient boosted regression tree (this only works on CPU)')
parser.add_argument('--graph', action='store_true', help='whether to use graph information')
parser.add_argument('--hetero', action='store_true', help='whether to treat as heterogeneous graph')
parser.add_argument('--dynamic', action='store_true', help='whether to use dynamic information')
parser.add_argument('--layer', type=int, default=2, help='number of layers')
parser.add_argument('--dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epoch', type=int, default=150, help='number of epochs')
parser.add_argument('--output', type=str, default='', help='save predictions to files')
args = parser.parse_args()

if args.data == 'setup6':
    args.num_snapshot = 100

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import dgl
import pickle
import xgboost
import time
import numpy as np
from minibatch import get_dataloader
from model import NetModel
from datetime import datetime
from sklearn import linear_model

t_inf = 0

if not args.gbrt and not args.sinr:
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader('data/{}/processed/'.format(args.data), args.batch_size, all_cuda=True)

    model = NetModel(args.layer, args.dim, args.graph, args.hetero, args.dynamic, args.num_snapshot).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def eval(model, dataloader, output=''):
        global t_inf
        if output != '':
            output_fn = 'output/{}.pkl'.format(output)
            if not os.path.exists(output_fn):
                os.makedirs(os.path.dirname(output_fn))
            outs = dict()
        model.eval()
        rmse = list()
        rmse_tot = 0
        with torch.no_grad():
            for g, idx in dataloader:
                t_s = time.time()
                g = g.to('cuda')
                mask = g.nodes['sta'].data['mask'] > 0
                pred = model(g)[mask]
                true = g.nodes['sta'].data['throughput'][mask]
                loss = torch.sqrt(torch.nn.functional.mse_loss(pred, true) + 1e-8)
                rmse.append(float(loss) * pred.shape[0])
                rmse_tot += pred.shape[0]
                t_inf += time.time() - t_s
                if output != '':
                    out = {'pred':pred.cpu().detach().numpy(), 'true':true.cpu().detach().numpy()}
                    outs[int(idx)] = out
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
elif args.gbrt:
    # GBRT using xgb library
    train_dataloader, _, test_dataloader = get_dataloader('data/{}/processed/'.format(args.data), args.batch_size, all_cuda=False)

    train_x = list()
    train_y = list()
    for g, _ in train_dataloader:
        mask = (g.nodes['sta'].data['mask'] > 0).squeeze()
        train_x.append(torch.cat([g.nodes['sta'].data['feat'], g.edges['sta_ap'].data['feat']], dim=1)[mask].numpy())
        train_y.append(g.nodes['sta'].data['throughput'][mask].numpy())
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=None)

    test_x = list()
    test_y = list()
    test_idx = list()
    test_length = list()
    for g, idx in test_dataloader:
        mask = (g.nodes['sta'].data['mask'] > 0).squeeze()
        test_x.append(torch.cat([g.nodes['sta'].data['feat'], g.edges['sta_ap'].data['feat']], dim=1)[mask].numpy())
        test_y.append(g.nodes['sta'].data['throughput'][mask].numpy())
        test_idx.append(int(idx))
        test_length.append(test_y[-1].shape[0])
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=None)
    
    model = xgboost.XGBRegressor(max_depth=4, n_estimators=100)
    t_s = time.time()
    model.fit(train_x, train_y)
    t_inf += time.time() - t_s
    pred = torch.from_numpy(model.predict(test_x))
    true = torch.from_numpy(test_y)
    rmse = float(torch.sqrt(torch.nn.functional.mse_loss(pred, true) + 1e-8))
    if args.output != '':
        output_fn = 'output/{}.pkl'.format(args.output)
        if not os.path.exists(output_fn):
            os.makedirs(os.path.dirname(output_fn))
        outs = dict()
        s_idx = 0
        e_idx = 0
        for idx, l in zip(test_idx, test_length):
            e_idx += l
            out = {'pred':pred[s_idx:e_idx].numpy(), 'true':true[s_idx:e_idx].numpy()}
            outs[idx] = out
            s_idx += l
        with open(output_fn, 'wb') as f:
            pickle.dump(outs, f)
    print('Test RMSE: {:.4f}'.format(rmse))
elif args.sinr:
    # linear regression using Scikit-learn
    train_dataloader, _, test_dataloader = get_dataloader('data/{}/processed/'.format(args.data), args.batch_size, all_cuda=False)

    train_x = list()
    train_y = list()
    for g, _ in train_dataloader:
        mask = (g.nodes['sta'].data['mask'] > 0).squeeze()
        train_x.append(torch.cat([g.nodes['sta'].data['feat'], g.edges['sta_ap'].data['feat']], dim=1)[mask].numpy())
        train_y.append(g.nodes['sta'].data['throughput'][mask].numpy())
    train_x = np.concatenate(train_x, axis=0)[:, 20].reshape(-1, 1)
    train_y = np.concatenate(train_y, axis=None)

    test_x = list()
    test_y = list()
    test_idx = list()
    test_length = list()
    for g, idx in test_dataloader:
        mask = (g.nodes['sta'].data['mask'] > 0).squeeze()
        test_x.append(torch.cat([g.nodes['sta'].data['feat'], g.edges['sta_ap'].data['feat']], dim=1)[mask].numpy())
        test_y.append(g.nodes['sta'].data['throughput'][mask].numpy())
        test_idx.append(int(idx))
        test_length.append(test_y[-1].shape[0])
    test_x = np.concatenate(test_x, axis=0)[:, 20].reshape(-1, 1)
    test_y = np.concatenate(test_y, axis=None)
    
    model = linear_model.LinearRegression()
    t_s = time.time()
    model.fit(train_x, train_y)
    t_inf += time.time() - t_s
    pred = torch.from_numpy(model.predict(test_x))
    true = torch.from_numpy(test_y)
    rmse = float(torch.sqrt(torch.nn.functional.mse_loss(pred, true) + 1e-8))
    if args.output != '':
        output_fn = 'output/{}.pkl'.format(args.output)
        if not os.path.exists(output_fn):
            os.makedirs(os.path.dirname(output_fn))
        outs = dict()
        s_idx = 0
        e_idx = 0
        for idx, l in zip(test_idx, test_length):
            e_idx += l
            out = {'pred':pred[s_idx:e_idx].numpy(), 'true':true[s_idx:e_idx].numpy()}
            outs[idx] = out
            s_idx += l
        with open(output_fn, 'wb') as f:
            pickle.dump(outs, f)
    print('Test RMSE: {:.4f}'.format(rmse))
print('Inference time per sequence: {:.4f}ms'.format(t_inf * 1000 / test_dataloader.__len__()))