import os
import pickle
import random
import sklearn.metrics as metrics
import numpy as np
from collections import defaultdict

for method in ['sinr', 'gbrt', 'mlp', 'gnn', 'mlp+lstm', 'gnn+lstm', 'htgnn']:
    latex = ''
    for setup in range(1, 7):
        d = pickle.load(open('output/main/setup{}_{}.pkl'.format(setup, method), 'rb'))
        true = list()
        pred = list()
        for k in d.keys():
            true.append(d[k]['true'])
            pred.append(d[k]['pred'])
        true = np.concatenate(true)
        pred = np.concatenate(pred)
        rmse = metrics.mean_squared_error(true, pred, squared=False)
        mae = metrics.mean_absolute_error(true, pred)
        print('setup{}_{:<8} RMSE:{:.4f} MAE:{:.4f}'.format(setup, method, rmse, mae))
        latex += ' & {:.4f} & {:.4f}'.format(rmse, mae)
    print(latex)
    print('')