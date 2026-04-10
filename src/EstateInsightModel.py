import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import os
import sys
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
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
    
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dataset = RealEstateDataset(TRAIN_DIR, transform=image_transform)
test_dataset = RealEstateDataset(TEST_DIR, transform=image_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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
        
        for name, param in self.model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True

    def forward(self, x):
        features = self.model(x)
        quality_output = self.quality_head(features)
        type_output = self.type_head(features)
        return quality_output, type_output

    
class EarlyStop:
    def __init__(self, patience = 10):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            self.early_stop = False
            return self.early_stop, True
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
            return self.early_stop, False

def train(dataloader, model, loss_fn, best_loss, optimizer, epoch, early_stop, device, writer):
    print()

    print(f"\n--- Training Epoch {epoch+1} ---")

    model.train()
    

    for batch, (x, quality_label, type_label) in enumerate(dataloader):
        x, quality_label, type_label = x.to(device), quality_label.to(device), type_label.to(device)
        pred_quality, pred_type = model(x)

        # convert integer labels to one-hot for BCEWithLogitsLoss
        quality_target = nn.functional.one_hot(quality_label, num_classes=model.quality_head.out_features).float()
        type_target = nn.functional.one_hot(type_label, num_classes=model.type_head.out_features).float()

        loss_quality = loss_fn(pred_quality, quality_target)  # compute loss for room quality
        loss_type = loss_fn(pred_type, type_target)  # compute loss for room type
        loss = loss_quality + loss_type

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.item(), batch)

        should_stop, is_improved = early_stop(loss.item())

        if is_improved:
            best_loss = loss
            print(f"New best model found! Loss: {loss.item():.6f} Saving...")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, MODEL_PATH)

        print(f"Batch {batch}: Loss = {loss.item():>7f} (Quality {loss_quality.item():.6f}, Type {loss_type.item():.6f})")

        if should_stop:
            return model, early_stop.best_loss, True
    
    print(f"Epoch {epoch+1} completed: {batch+1} batches processed")
    return model, best_loss, False

def evaluate(dataloader, model, loss_fn, device, writer):
    print()
    print("--- Eval Model ---")
    test_loss = 0.0
    correct_quality = 0
    correct_type = 0
    correct_both = 0
    total = 0

    model.eval()

    with torch.no_grad():
        batch_count = 0
        for batch, (x, quality_label, type_label) in enumerate(dataloader):
            x, quality_label, type_label = x.to(device), quality_label.to(device), type_label.to(device)
            pred_quality, pred_type = model(x)

            batch_size = quality_label.size(0)
            total += batch_size
            batch_count += 1

            # convert to one-hot for BCEWithLogitsLoss
            quality_target = nn.functional.one_hot(quality_label, num_classes=model.quality_head.out_features).float()
            type_target = nn.functional.one_hot(type_label, num_classes=model.type_head.out_features).float()

            loss_q = loss_fn(pred_quality, quality_target)
            loss_t = loss_fn(pred_type, type_target)
            test_loss += (loss_q + loss_t)
            print("Batch {}: Loss = {:.6f} (Quality {:.6f}, Type {:.6f})".format(batch, loss_q + loss_t, loss_q, loss_t))

            # predictions
            q_preds = pred_quality.argmax(1)
            t_preds = pred_type.argmax(1)

            correct_quality += int((q_preds == quality_label).type(torch.long).sum().item())
            correct_type += int((t_preds == type_label).type(torch.long).sum().item())
            correct_both += int(((q_preds == quality_label) & (t_preds == type_label)).type(torch.long).sum().item())

    writer.add_scalar("Loss/test", test_loss / total)

    print("Total Samples: ", total)
    print("Correct Quality Predictions: ", correct_quality)
    print("Correct Type Predictions: ", correct_type)
    print("Correct Both (quality + type): ", correct_both)
    print(f"Test Loss: {test_loss / max(total,1):.4f}")
    print(f"Evaluation: Quality Accuracy = {int(100 * correct_quality / max(total,1))}%")
    print(f"Evaluation: Type Accuracy = {int(100 * correct_type / max(total,1))}%")
    print(f"Evaluation: Combined Accuracy = {int(100 * correct_both / max(total,1))}%")

    return test_loss / batch_count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)

    print()
    print("--- Tensorboard Setup---")
    writer = SummaryWriter(LOG_DIR)

    print()
    print("--- Instantiate Model ---")
    model = EstateInsightModel(num_quality_classes=3, num_type_classes=5).to(device)
    # model = PreTrainedModel().to(device)
    best_loss = float('inf')
    

    NUM_EPOCHS = 50
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.002
    )
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stop = EarlyStop() # patience = 10

    early_stop = EarlyStop(150) # patience = 20 by default

    print("--- Load Best Model ---")
    if os.path.exists(MODEL_PATH):
        best_model = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model['model_state_dict'])
        optimizer.load_state_dict(best_model['optimizer_state_dict'])
        best_loss = best_model['loss']
        print("Loaded best model from ", MODEL_PATH)

    for epoch in range(NUM_EPOCHS):
        model, best_loss, should_stop = train(train_loader, model, criterion, best_loss, optimizer, epoch, early_stop, device, writer)
        test_loss = evaluate(test_loader, model, criterion, device, writer)
        scheduler.step()

        if should_stop:
            print(f"Model has stopped training after {epoch+1} epochs")
            break

    writer.close()

if __name__ == "__main__":
    main()