import os
import os.path as osp
import shutil
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data import Data, DataLoader


url = ['https://ndownloader.figshare.com/files/25605734', 'https://ndownloader.figshare.com/files/25605737', 'https://ndownloader.figshare.com/files/25605740']
#url = ['https://ndownloader.figshare.com/files/25605740']

class COLL(InMemoryDataset):
    def __init__(self, root='dataset/', subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        self.folder = osp.join(root, 'coll')
        assert split in ['train', 'val', 'test']
        super(COLL, self).__init__(self.folder, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        #return ['train.xyz', 'val.xyz', 'test.xyz']
        return ['coll_v1.2_AE_train.xyz', 'coll_v1.2_AE_val.xyz', 'coll_v1.2_AE_test.xyz']

    @property
    def raw_dir(self):
        return osp.join(self.folder, 'raw')

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']


    def download(self):
        for u in url:
            path = download_url(u, self.raw_dir)
            os.unlink(path)

    #def molecule(self, num_nodes, Z, pos, forces, pbc, energy, atomization_energy):
    #def parser(self, name):

    def process(self):
            #R_i = torch.tensor(R_qm9[i],dtype=torch.float32)
            #z_i = torch.tensor(Z_qm9[i],dtype=torch.int64)
            #y_i = [torch.tensor(target[name][i],dtype=torch.float32) for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']]
            #data = Data(pos=R_i, z=z_i, y=y_i[0], mu=y_i[0], alpha=y_i[1], homo=y_i[2], lumo=y_i[3], gap=y_i[4], r2=y_i[5], zpve=y_i[6], U0=y_i[7], U=y_i[8], H=y_i[9], G=y_i[10], Cv=y_i[11])
        for idx in range(len(self.raw_file_names)):
            name = osp.join(self.raw_dir, self.raw_file_names[idx])
            data_list = []
            with open(name, encoding='utf-8') as f:
                mol_split = 0

                for line in f:
                    object = line.rstrip()

                    if object.find(' ') == -1:
                        num_nodes = 0
                        z_i = []
                        pos_i = []
                        forces_i = []
                        num_nodes = int(object)

                        mol_split = num_nodes + 1
                        continue

                    if mol_split == num_nodes + 1:
                        energy = torch.tensor(float(object.split()[-2].split('=')[-1]), dtype=torch.float32)
                        atomization_energy = torch.tensor(float(object.split()[-1].split('=')[-1]), dtype=torch.float32)
                    else:
                        object = object.split()
                        z_i.append(torch.tensor([ord(i) for i in object[0]], dtype=torch.int8))
                        pos_i.append(torch.tensor([float(i) for i in object[1:4]], dtype=torch.float32))
                        forces_i.append(torch.tensor([float(i) for i in object[4:]], dtype=torch.float32))

                    mol_split -= 1
                    if mol_split == 0:
                        data = Data(pos=pos_i, z=z_i, forces=forces_i, energy=energy, atomization_energy=atomization_energy, num_nodes=num_nodes)
                        data_list.append(data)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)

            print('Saving...')
            torch.save((data, slices), self.processed_paths[idx])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

if __name__ == '__main__':
    dataset = COLL()
    #print(dataset)
    #print(dataset.data.z.shape)
    #print(dataset.data.pos.shape)
    #target = 'mu'
    #dataset.data.y = dataset.data[target]
    #print(dataset.data.y.shape)
    #print(dataset.data.y)
    #print(dataset.data.mu)
    #split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
    #print(split_idx)
    #print(dataset[split_idx['train']])
    #train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #data = next(iter(train_loader))
    #print(data)
