import xlwt
import os

book = xlwt.Workbook()
sheet = book.add_sheet('layer')

def get_rmse(fn):
    rmse = 0
    with open(fn, 'r') as f:
        for l in f:
            if l.startswith('Test RMSE'):
                rmse = float(l.split(':')[1])
    return rmse

col = 0
for setup in range(1, 7):
    for method in ['gnn', 'gnn+lstm', 'htgnn']:
        sheet.write(0, col, '{}{}'.format(setup, method))
        for l in range(1, 13):
            rmse = get_rmse('output/layer/setup{}/{}_{}.out'.format(setup, method, l))
            sheet.write(l, col, rmse)
        col += 1

runtime = dict()
with open('output/layer/runtime.out', 'r') as f:
    exp = ''
    for l in f:
        if l.startswith('setup'):
            exp = l.strip()
        elif l.startswith('Inference time'):
            runtime[exp] = float(l.split(':')[1].split('ms')[0])
for setup in range(1, 7):
    for method in ['gnn', 'gnn+lstm', 'htgnn']:
        sheet.write(0, col, 't{}{}'.format(setup, method))
        for l in range(1, 13):
            sheet.write(l, col, runtime['setup{}_{}_{}'.format(setup, method, l)])
        col += 1

book.save('output/layer/layer.xls')
