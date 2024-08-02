from torch import nn
from torch_points3d.applications.pointnet import PointNet2  # type: ignore


class PointNetWrapper(nn.Module):
    def __init__(self, num_classes, feat_size):
        super().__init__()
        self.pointnet = PointNet2({"feat_size": feat_size, "n_classes": num_classes})

    def forward(self, x):
        x = self.pointnet(x)
        return x
