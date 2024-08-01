from torchvision import models


# RCNN for 2D detections
# CountNet is agnostic to this model
class ObjectDetector2D:
    def __init__(self):
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    def forward(self, images):
        detections = self.model(images)
        return detections
