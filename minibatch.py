import os
import torch
import dgl

class NetworkData(torch.utils.data.Dataset):

    def __init__(self, fn_dir):
        train = list()
        valid = list()
        test = list()
        for fn in os.listdir(fn_dir):
            if fn.startswith('train'):
                train.append(dgl.load_graphs(fn_dir + fn)[0][0])
            elif fn.startswith('valid'):
                valid.append(dgl.load_graphs(fn_dir + fn)[0][0])
            else:
                test.append(dgl.load_graphs(fn_dir + fn)[0][0])
        self.data = train + valid + test
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(torch.arange(len(train)))
        self.valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(torch.arange(len(train), len(train) + len(valid)))
        self.test_sampler = torch.utils.data.sampler.SubsetRandomSampler(torch.arange(len(train) + len(valid), len(train) + len(valid) + len(test)))

    def __len__(self):
        return len(Self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def to_cuda(self):
        cuda_data = [d.to('cuda') for d in self.data]
        self.data = cuda_data

    def get_sampler(self):
        return self.train_sampler, self.valid_sampler, self.test_sampler

def get_dataloader(fn_dir, batchsize, all_cuda=False):
    print('loading data...')
    dataset = NetworkData(fn_dir)
    if all_cuda:
        dataset.to_cuda()
    train_sampler, valid_sampler, test_sampler = dataset.get_sampler()

    train_dataloader = dgl.dataloading.GraphDataLoader(dataset, sampler=train_sampler, batch_size=batchsize)
    valid_dataloader = dgl.dataloading.GraphDataLoader(dataset, sampler=valid_sampler, batch_size=batchsize)
    test_dataloader = dgl.dataloading.GraphDataLoader(dataset, sampler=test_sampler, batch_size=batchsize)

    return train_dataloader, valid_dataloader, test_dataloader