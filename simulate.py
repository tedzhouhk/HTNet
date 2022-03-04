import glob
import os
from subprocess import Popen, DEVNULL
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--setup', type=int, help='which setup to simulate')
args = parser.parse_args()

p = Popen('find data/setup{}/raw/ -name "*.out" -delete'.format(args.setup), shell=True)
p.wait()
print(p.returncode)

flist = list()
for file in os.listdir('data/setup{}/raw'.format(args.setup)):
    if file.endswith('.csv'):
        flist.append('data/setup{}/raw/'.format(args.setup) + file)
flist.sort()

shell_command="komondor_main {} {} sim_{} 0 0 1 10 1992"

p_list = list()
for f in tqdm(flist):
    cmd = shell_command.format(f, f[:-3] + 'out', f.split('/')[-1].split('.')[0])
    if len(p_list) < 64:
        p_list.append(Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL))
    else:
        for p in p_list:
            p.wait()
            p.kill()
        p_list = list()
        p_list.append(Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL))
for p in p_list:
    p.wait()
    p.kill()