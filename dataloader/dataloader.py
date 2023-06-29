import torch
from torch.utils.data import DataLoader, Dataset
from dataloader.textloader import load_preprocessing_data


class ToxicCmtDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).to(torch.long)
        self.y = torch.from_numpy(y).to(torch.int)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def loader_dataset(df):
    x, y = load_preprocessing_data(df)
    dataset = ToxicCmtDataset(x, y)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    return dataloader
