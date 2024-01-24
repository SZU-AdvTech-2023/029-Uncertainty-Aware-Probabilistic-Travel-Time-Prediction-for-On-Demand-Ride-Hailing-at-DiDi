import random, time
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TrajDataset(Dataset):
    def __init__(self, file):
        print(f"Loading data from {file}...")
        a = time.time()
        self.data = pd.read_csv(file, sep=";", encoding="utf8",
                                converters={"LinkIDs": eval, "LinkDistances": eval})

        self.label1 = self.data[["Duration"]].values.astype(float).tolist()
        self.dense = self.data[["Distance"]].values.tolist()
        self.sparse = self.data[["Day", "Time"]].values.tolist()

        self.seq_dense = []
        self.seq_sparse = []
        self.link_len = [16] * len(self.data)

        sds = self.data[["LinkDistances"]].values

        for sd in sds:
            t = np.array(sd.tolist()).T
            self.seq_dense.append(torch.tensor(t.tolist()))
            # self.link_len.append(t.shape[0])

        sss = self.data[["LinkIDs"]].values
        for ss in sss:
            t = np.array(ss.tolist()).T
            self.seq_sparse.append(torch.tensor(t.tolist()))

        b = time.time()
        print(f"Finished! ({round(b - a) // 60}:{round(b - a) % 60})")

    def __getitem__(self, indices):
        if isinstance(indices, int):
            dense = torch.tensor([self.dense[indices]])
            sparse = torch.tensor([self.sparse[indices]])
            # (batch_size, seq_len, feature)
            seq_dense = [self.seq_dense[indices]]
            seq_sparse = [self.seq_sparse[indices]]
            seq_dense = pad_sequence(seq_dense, batch_first=True)
            seq_sparse = pad_sequence(seq_sparse, batch_first=True)
            link_len = torch.tensor([self.link_len[indices]])
            label = torch.tensor([self.label1[indices]])
            return (dense, sparse, seq_dense, seq_sparse), link_len, label
        elif isinstance(indices, (list, tuple)):
            dense = torch.tensor([self.dense[i] for i in indices])
            sparse = torch.tensor([self.sparse[i] for i in indices])
            # (batch_size, seq_len, feature)
            seq_dense = [self.seq_dense[i] for i in indices]
            seq_sparse = [self.seq_sparse[i] for i in indices]
            seq_dense = pad_sequence(seq_dense, batch_first=True)
            seq_sparse = pad_sequence(seq_sparse, batch_first=True)
            link_len = torch.tensor([self.link_len[i] for i in indices])
            label = torch.tensor([self.label1[i] for i in indices])
            return (dense, sparse, seq_dense, seq_sparse), link_len, label

    def __len__(self):
        return len(self.data)


class TrajDataset2(Dataset):
    def __init__(self, file):
        print(f"Loading data from {file}...")
        a = time.time()
        self.data = pd.read_csv(file, sep=";", encoding="utf8",
                                converters={"LinkIDs": eval, "LinkDistances": eval})

        self.label1 = self.data[["Duration"]].values.astype(float).tolist()
        self.label2 = self.data[["C_duration"]].values.tolist()
        self.dense = self.data[["Distance"]].values.tolist()
        self.sparse = self.data[["Day", "Time"]].values.tolist()

        self.seq_dense = []
        self.seq_sparse = []
        self.link_len = [16] * len(self.data)

        sds = self.data[["LinkDistances"]].values

        for sd in sds:
            t = np.array(sd.tolist()).T
            self.seq_dense.append(torch.tensor(t.tolist()))
            # self.link_len.append(t.shape[0])

        sss = self.data[["LinkIDs"]].values
        for ss in sss:
            t = np.array(ss.tolist()).T
            self.seq_sparse.append(torch.tensor(t.tolist()))

        b = time.time()
        print(f"Finished! ({round(b - a) // 60}:{round(b - a) % 60})")

    def __getitem__(self, indices):
        if isinstance(indices, int):
            dense = torch.tensor([self.dense[indices]])
            sparse = torch.tensor([self.sparse[indices]])
            # (batch_size, seq_len, feature)
            seq_dense = [self.seq_dense[indices]]
            seq_sparse = [self.seq_sparse[indices]]
            seq_dense = pad_sequence(seq_dense, batch_first=True)
            seq_sparse = pad_sequence(seq_sparse, batch_first=True)
            link_len = torch.tensor([self.link_len[indices]])
            label1 = torch.tensor([self.label1[indices]])
            label2 = torch.tensor([self.label2[indices]])
            return (dense, sparse, seq_dense, seq_sparse), link_len, (label1, label2)
        elif isinstance(indices, (list, tuple)):
            dense = torch.tensor([self.dense[i] for i in indices])
            sparse = torch.tensor([self.sparse[i] for i in indices])
            # (batch_size, seq_len, feature)
            seq_dense = [self.seq_dense[i] for i in indices]
            seq_sparse = [self.seq_sparse[i] for i in indices]
            seq_dense = pad_sequence(seq_dense, batch_first=True)
            seq_sparse = pad_sequence(seq_sparse, batch_first=True)
            link_len = torch.tensor([self.link_len[i] for i in indices])
            label1 = torch.tensor([self.label1[i] for i in indices])
            label2 = torch.tensor([self.label2[i] for i in indices])
            return (dense, sparse, seq_dense, seq_sparse), link_len, (label1, label2)

    def __len__(self):
        return len(self.data)


class TrajDataset3(Dataset):
    def __init__(self, file):
        print(f"Loading data from {file}...")
        a = time.time()
        self.data = pd.read_csv(file, sep=";", encoding="utf8",
                                converters={"LinkIDs": eval, "LinkDistances": eval})

        self.label1 = self.data[["Duration"]].values.astype(float).tolist()
        self.label2 = self.data[["C_duration"]].values.tolist()
        self.dense = self.data[["Distance"]].values.tolist()
        self.sparse = self.data[["Day", "Time"]].values.tolist()

        self.seq_dense = []
        self.seq_sparse = []
        self.link_len = [16] * len(self.data)

        sds = self.data[["LinkDistances"]].values

        for sd in sds:
            t = np.array(sd.tolist()).T
            self.seq_dense.append(torch.tensor(t.tolist()))
            # self.link_len.append(t.shape[0])

        sss = self.data[["LinkIDs"]].values
        for ss in sss:
            t = np.array(ss.tolist()).T
            self.seq_sparse.append(torch.tensor(t.tolist()))

        b = time.time()
        print(f"Finished! ({round(b - a) // 60}:{round(b - a) % 60})")

    def __getitem__(self, indices):
        if isinstance(indices, int):
            dense = torch.tensor([self.dense[indices]])
            sparse = torch.tensor([self.sparse[indices]])
            # (batch_size, seq_len, feature)
            seq_dense = [self.seq_dense[indices]]
            seq_sparse = [self.seq_sparse[indices]]
            seq_dense = pad_sequence(seq_dense, batch_first=True)
            seq_sparse = pad_sequence(seq_sparse, batch_first=True)
            link_len = torch.tensor([self.link_len[indices]])
            label1 = torch.tensor([self.label1[indices]])
            # label2 = torch.tensor([self.label2[indices]])
            label2 = torch.zeros((1, 5))
            if self.label2[indices][0] == 0:
                label2[0, 0] = 0.6
                label2[0, 1] = 0.4
            elif self.label2[indices][0] == 1:
                label2[0, 0] = 0.2
                label2[0, 1] = 0.6
                label2[0, 2] = 0.2
            elif self.label2[indices][0] == 2:
                label2[0, 1] = 0.2
                label2[0, 2] = 0.6
                label2[0, 3] = 0.2
            elif self.label2[indices][0] == 3:
                label2[0, 2] = 0.2
                label2[0, 3] = 0.6
                label2[0, 4] = 0.2
            elif self.label2[indices][0] == 4:
                label2[0, 3] = 0.2
                label2[0, 4] = 0.6
            return (dense, sparse, seq_dense, seq_sparse), link_len, (label1, label2)
        elif isinstance(indices, (list, tuple)):
            dense = torch.tensor([self.dense[i] for i in indices])
            sparse = torch.tensor([self.sparse[i] for i in indices])
            # (batch_size, seq_len, feature)
            seq_dense = [self.seq_dense[i] for i in indices]
            seq_sparse = [self.seq_sparse[i] for i in indices]
            seq_dense = pad_sequence(seq_dense, batch_first=True)
            seq_sparse = pad_sequence(seq_sparse, batch_first=True)
            link_len = torch.tensor([self.link_len[i] for i in indices])
            label1 = torch.tensor([self.label1[i] for i in indices])
            # label2 = torch.tensor([self.label2[i] for i in indices])
            label2 = torch.zeros((len(indices), 5))
            for i in range(len(indices)):
                if self.label2[indices[i]][0] == 0:
                    label2[i, 0] = 0.6
                    label2[i, 1] = 0.4
                elif self.label2[indices[i]][0] == 1:
                    label2[i, 0] = 0.2
                    label2[i, 1] = 0.6
                    label2[i, 2] = 0.2
                elif self.label2[indices[i]][0] == 2:
                    label2[i, 1] = 0.2
                    label2[i, 2] = 0.6
                    label2[i, 3] = 0.2
                elif self.label2[indices[i]][0] == 3:
                    label2[i, 2] = 0.2
                    label2[i, 3] = 0.6
                    label2[i, 4] = 0.2
                elif self.label2[indices[i]][0] == 4:
                    label2[i, 3] = 0.2
                    label2[i, 4] = 0.6
            return (dense, sparse, seq_dense, seq_sparse), link_len, (label1, label2)

    def __len__(self):
        return len(self.data)

def traj_dataloader(dataset: [TrajDataset, TrajDataset2],
                    batch_size: int, shuffle: bool):
    num_sample = len(dataset)
    indices = list(range(num_sample))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, num_sample, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_sample)]
        yield dataset[batch_indices]
