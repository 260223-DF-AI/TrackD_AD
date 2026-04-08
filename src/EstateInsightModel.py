import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import os
import sys
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from data_loader import RealEstateDataset


DATA_ROOT = "src/data/"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LOG_DIR = "runs/estate_insight_logs"
MODEL_PATH = "estate_insight.pth"

if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Dataset directory '{TRAIN_DIR}' not found.")
    print("Please ensure the 'estate_insight' folder is extracted in the script's directory.")
    sys.exit(1)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

train_dataset = RealEstateDataset(TRAIN_DIR, transform=image_transform)
test_dataset = RealEstateDataset(TEST_DIR, transform=image_transform)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

num_quality_classes = len(train_dataset.quality_label_map)
num_type_classes = len(train_dataset.section_label_map)

print(f"Number of quality classes: {num_quality_classes}")
print(f"Number of type classes: {num_type_classes}")
print(f"Total training images available: {len(train_dataset)}")

class EstateInsightModel(nn.Module):
    def __init__(self, num_quality_classes, num_type_classes):
        super(EstateInsightModel, self).__init__()
        # Load a pre-trained ResNet model
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Replace the final fully connected layer to match the number of classes
        in_features = self.model.fc.in_features

        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # remove original fully connected layer
        self.model.fc = nn.Identity()

        # Two separate outputs for the quality and type classifications
        self.quality_head = nn.Linear(in_features, num_quality_classes)
        self.type_head = nn.Linear(in_features, num_type_classes)

    def forward(self, x):
        features = self.model(x)
        quality_output = self.quality_head(features)
        type_output = self.type_head(features)
        return quality_output, type_output

def train(dataloader, model, loss_fn, optimizer, epoch, device):
    print()

    print(f"\n--- Training Epoch {epoch+1} ---")

    model.train()
    
    for batch, (x, quality_label, type_label) in enumerate(dataloader):
        x, quality_label, type_label = x.to(device), quality_label.to(device), type_label.to(device)
        pred_quality, pred_type = model(x)
        loss_quality = loss_fn(pred_quality, quality_label)
        loss_type = loss_fn(pred_type, type_label)
        loss = loss_quality + loss_type

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch {batch}: Loss = {loss.item():>7f}")
    
    print(f"Epoch {epoch+1} completed: {batch+1} batches processed")
    return model

def evaluate(dataloader, model, loss_fn, device):
    print()
    print("--- Eval Model ---")
    test_loss = 0.0
    correct_quality = 0
    correct_type = 0
    correct_both = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for batch, (x, quality_label, type_label) in enumerate(dataloader):
            x, quality_label, type_label = x.to(device), quality_label.to(device), type_label.to(device)
            pred_quality, pred_type = model(x)

            batch_size = quality_label.size(0)
            total += batch_size

            # accumulate loss for both heads
            loss_q = loss_fn(pred_quality, quality_label).item()
            loss_t = loss_fn(pred_type, type_label).item()
            test_loss += (loss_q + loss_t)

            # predictions
            q_preds = pred_quality.argmax(1)
            t_preds = pred_type.argmax(1)

            correct_quality += int((q_preds == quality_label).type(torch.long).sum().item())
            correct_type += int((t_preds == type_label).type(torch.long).sum().item())
            correct_both += int(((q_preds == quality_label) & (t_preds == type_label)).type(torch.long).sum().item())

            if batch == 9:
                break

    print("Total Samples: ", total)
    print("Correct Quality Predictions: ", correct_quality)
    print("Correct Type Predictions: ", correct_type)
    print("Correct Both (quality + type): ", correct_both)
    print(f"Test Loss: {test_loss / max(total,1):.4f}")
    print(f"Evaluation: Quality Accuracy = {int(100 * correct_quality / max(total,1))}%")
    print(f"Evaluation: Type Accuracy = {int(100 * correct_type / max(total,1))}%")
    print(f"Evaluation: Combined Accuracy = {int(100 * correct_both / max(total,1))}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)

    print()
    print("--- Instantiate Model ---")
    model = EstateInsightModel(num_quality_classes=3, num_type_classes=5).to(device)
    # model = PreTrainedModel().to(device)
    best_loss = float('inf')
    

    NUM_EPOCHS = 1
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    print("--- Load Best Model ---")
    if os.path.exists(MODEL_PATH):
        best_model = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model['model_state_dict'])
        optimizer.load_state_dict(best_model['optimizer_state_dict'])
        best_loss = best_model['loss']
        print("Loaded best model from ", MODEL_PATH)

    for epoch in range(NUM_EPOCHS):
        model = train(train_loader, model, criterion, optimizer, epoch, device)
        evaluate(test_loader, model, criterion, device)

if __name__ == "__main__":
    main()