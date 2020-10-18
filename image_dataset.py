import torch.utils.data.Dataset

class ImageDirectoryDataset(Dataset):
    def __init__(self, imageDir, label)
        self.imageDir = imageDir
        self.label = label
    def __len__(self):

    def __getitem__(self, idx):
