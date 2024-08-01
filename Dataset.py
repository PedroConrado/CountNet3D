import torch
import torch.nn.functional as F  # noqa: N812, F401
from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, 640, 480)
        point_cloud = torch.randn(1024, 3)
        count = torch.tensor([idx % 10], dtype=torch.float32)  # Exemplo de contagem
        return image, point_cloud, count
