import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class EstateInsightModel(nn.Module):
    def __init__(self, num_classes):
        super(EstateInsightModel, self).__init__()
        # Load a pre-trained ResNet model
        self.resnet = models.resnet18(pretrained=True, weights=models.ResNet18_Weights.DEFAULT)
        # Replace the final fully connected layer to match the number of classes
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

