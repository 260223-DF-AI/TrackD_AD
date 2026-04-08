import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from .data_loader import RealEstateDataset, image_transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time

DATA_DIR = 'data/'
image_dataset = RealEstateDataset(DATA_DIR, image_transform)
house_loader = DataLoader(image_dataset, batch_size=10, shuffle=True)


class EstateInsightModel(nn.Module):
    """Resnet18 Model. Fine-tuned by freezing all layers, except fully connected layers for both label classes. Uses Softmax activated function"""
    
    def __init__(self):
        super(EstateInsightModel, self).__init__()
        
        # Load a pre-trained ResNet resnet_model
        self.resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        #Freeze all layers
        for param in self.resnet_model.parameters:
            param.requires_grad = False
            
        # Get input features from the last layer
        in_features = self.resnet_model.fc.in_features
        
        # Replace original fully connected layer with a container. nn.Identity() just returned input  is used to remove layers when used like this
        self.resnet_model.fc = nn.Identity()
        
        # Create two separate layers for each label
        num_qualities = 3
        self.quality_labels = nn.Linear(in_features, num_qualities)
        
        num_sections = 5
        self.section_labels = nn.Linear(in_features, num_sections)
        
        
        num_qualities = 3
        
    def forward(self, x):
        # Extract features
        features = self.resnet_model(x)

        # Pass features through both layers for House Section (Room) and Quality labels
        quality_out = self.quality_labels(features)
        section_out = self.section_labels(features)
        
        return quality_out, section_out
    
class EarlyStop:
    def __init__(self, patience = 100):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return self.early_stop
        else:
            self.count += 1
            if self.counter > self.patience:
                self.early_stop = True
        return self.early_stop
    
def train_model(dataloader, model, quality_loss_fn, section_loss_fn, optimizer, epochs, best_loss, writer, device, early_stop):
    
    model.to(device)
    model.train()
    start_time = time.time()
    for epoch in epochs:
        running_loss = 0.0
        for batch_idx, (images, quality_labels, section_labels) in enumerate(dataloader):
            images, quality_labels, section_labels = images.to(device), quality_labels.to(device), section_labels.to(device)
            optimizer.zero_grad()
            quality_pred, section_pred = model(images)
            
            # Calculate invidual loss
            loss_quality = quality_loss_fn(quality_pred, quality_labels)
            loss_section = section_loss_fn(section_pred, section_labels)

            # Combined loss
            total_loss = loss_quality + loss_section
            
            # Backwards pass
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")
            
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
if __name__ == "__main__":
    main()
    