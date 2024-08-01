import torch
import torch.nn.functional as F  # noqa: N812
from ObjectDetector import ObjectDetector2D
from PointNet import PointNetWrapper
from torch import nn


# CountNet3D para contagem de objetos
class CountNet3D(nn.Module):
    def __init__(self, num_classes, geometry_dict):
        super().__init__()
        self.yolo = ObjectDetector2D()
        self.pointnet = PointNetWrapper(num_classes)
        self.fc1 = nn.Linear(1024 + len(geometry_dict) + 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.geometry_dict = geometry_dict

    def forward(self, images, point_clouds):
        detections = self.yolo(images)
        point_beams, geometry_labels = self.create_point_beams(detections, point_clouds)
        counts = []
        for point_beam, geometry_label in zip(point_beams, geometry_labels, strict=False):
            x = self.pointnet(point_beam)
            geometry_one_hot = self.geometry_to_one_hot(geometry_label).unsqueeze(0)
            x = torch.cat([x, geometry_one_hot.repeat(x.size(0), 1)], dim=1)
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            count = self.fc3(x)
            counts.append(count)
        return torch.stack(counts)

    def create_point_beams(self, detections, point_clouds):
        point_beams = []
        geometry_labels = []
        for detection, point_cloud in zip(detections, point_clouds, strict=False):
            for box in detection["boxes"]:
                point_beam = self.lift_2d_to_3d(box, point_cloud)
                point_beam = self.orthogonal_rotation(point_beam)
                point_beam = self.mean_shift(point_beam)
                depth_features = self.beam_depth(point_beam)
                point_beam = torch.cat([point_beam, depth_features], dim=1)
                point_beams.append(point_beam)
                geometry_labels.append(self.get_geometry_label(detection))
        return point_beams, geometry_labels

    def lift_2d_to_3d(self, box, point_cloud):
        x_min, y_min, x_max, y_max = box

        # Propriedades intrínsecas e extrínsecas da câmera (exemplo)
        fx, fy = 525.0, 525.0
        cx, cy = 319.5, 239.5
        extrinsic_matrix = torch.eye(4)

        points_3d = point_cloud[:, :3]
        ones = torch.ones(points_3d.size(0), 1)
        points_3d_hom = torch.cat([points_3d, ones], dim=1)

        points_2d_hom = torch.mm(points_3d_hom, extrinsic_matrix.T)
        points_2d_hom = points_2d_hom[:, :3]
        points_2d_hom[:, 0] = points_2d_hom[:, 0] / points_2d_hom[:, 2]
        points_2d_hom[:, 1] = points_2d_hom[:, 1] / points_2d_hom[:, 2]

        u = fx * points_2d_hom[:, 0] + cx
        v = fy * points_2d_hom[:, 1] + cy

        mask = (u >= x_min) & (u <= x_max) & (v >= y_min) & (v <= y_max)
        point_beam = points_3d[mask]

        return point_beam

    def orthogonal_rotation(self, point_cloud):
        z = point_cloud[:, 2]
        angle = torch.atan2(z, point_cloud[:, 0])
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        rot_matrix = torch.stack([cos_angle, -sin_angle, sin_angle, cos_angle], dim=1).reshape(-1, 2, 2)
        rotated_points = torch.bmm(rot_matrix, point_cloud[:, :2].unsqueeze(2)).squeeze(2)
        return torch.cat([rotated_points, point_cloud[:, 2:]], dim=1)

    def mean_shift(self, point_cloud):
        mean = torch.mean(point_cloud, dim=0, keepdim=True)
        shifted_points = point_cloud - mean
        return shifted_points

    def beam_depth(self, point_cloud):
        ymin = torch.min(point_cloud[:, 1])
        ymax = torch.max(point_cloud[:, 1])
        depth_features = torch.stack([ymax - point_cloud[:, 1], point_cloud[:, 1] - ymin], dim=1)
        return depth_features

    def get_geometry_label(self, detection):
        label = detection["labels"][0]
        return self.geometry_dict.get(label.item(), len(self.geometry_dict))

    def geometry_to_one_hot(self, geometry_label):
        one_hot = torch.zeros(len(self.geometry_dict))
        one_hot[geometry_label] = 1
        return one_hot


