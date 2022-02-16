import glob
import os
from subprocess import Popen, DEVNULL
from tqdm import tqdm

p = Popen('find data/setup1/raw/ -name "*.out" -delete', shell=True)
p.wait()
print(p.returncode)

flist = list()
for file in os.listdir('data/setup1/raw'):
    if file.endswith('.csv'):
        flist.append('data/setup1/raw/' + file)
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