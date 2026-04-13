import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import os
import sys
import logging
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from data_loader import RealEstateDataset
from torchmetrics import Accuracy, Precision, Recall, F1Score

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create file handler for reporting
file_handler = logging.FileHandler('reporting.log')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger (only if not already added)
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

num_quality_classes = len(train_dataset.quality_label_map)
num_type_classes = len(train_dataset.section_label_map)

print(f"Number of quality classes: {num_quality_classes}")
print(f"Number of type classes: {num_type_classes}")
print(f"Total training images available: {len(train_dataset)}")

class EstateInsightModel(nn.Module):
    """
    Custom model for real estate image classification that predicts both room quality and type.
    Uses a pre-trained ResNet backbone with separate heads for each classification task.
    The third and fourth layers of ResNet are unfrozen for fine-tuning, while earlier layers are frozen to retain general features.
    """
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
            if "layer4" in name or "fc" in name or "layer3" in name:
                param.requires_grad = True

    def forward(self, x):
        features = self.model(x)
        quality_output = self.quality_head(features)
        type_output = self.type_head(features)
        return quality_output, type_output

    def predict_with_confidence(self, x):
        """
        Get predictions with confidence scores for both quality and type.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            dict: Contains:
                - quality_preds: Predicted quality class indices (batch_size,)
                - quality_confidence: Confidence scores for quality predictions (batch_size,)
                - type_preds: Predicted type class indices (batch_size,)
                - type_confidence: Confidence scores for type predictions (batch_size,)
                - quality_probs: Full probability distributions for quality (batch_size, num_classes)
                - type_probs: Full probability distributions for type (batch_size, num_classes)
        """
        with torch.no_grad():
            quality_logits, type_logits = self.forward(x)
            
            # Apply softmax to get probabilities
            quality_probs = torch.softmax(quality_logits, dim=1)
            type_probs = torch.softmax(type_logits, dim=1)
            
            # Get max probability (confidence) and corresponding class indices
            quality_confidence, quality_preds = torch.max(quality_probs, dim=1)
            type_confidence, type_preds = torch.max(type_probs, dim=1)
            
            return {
                'quality_preds': quality_preds,
                'quality_confidence': quality_confidence,
                'type_preds': type_preds,
                'type_confidence': type_confidence,
                'quality_probs': quality_probs,
                'type_probs': type_probs
            }

    
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
    total_loss = 0
    batch_count = 0

    for batch, (x, quality_label, type_label) in enumerate(dataloader):
        x, quality_label, type_label = x.to(device), quality_label.to(device), type_label.to(device)
        pred_quality, pred_type = model(x)

        # compute loss for room quality and type
        loss_quality = loss_fn(pred_quality, quality_label)
        loss_type = loss_fn(pred_type, type_label)
        loss = loss_quality + loss_type

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.item(), batch)
        total_loss += loss.item()
        batch_count += 1

        should_stop, is_improved = early_stop(loss.item())

        if is_improved:
            best_loss = loss
            print(f"New best model found! Loss: {loss.item():.6f} Saving...")
            logger.info(f"Epoch {epoch+1}: New best model found with loss {loss.item():.6f}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, MODEL_PATH)

        print(f"Batch {batch}: Loss = {loss.item():>7f} (Quality {loss_quality.item():.6f}, Type {loss_type.item():.6f})")

        if should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            return model, early_stop.best_loss, True
    
    avg_epoch_loss = total_loss / max(batch_count, 1)
    writer.add_scalar("Loss/train_epoch_avg", avg_epoch_loss, epoch)
    print(f"Epoch {epoch+1} completed: {batch_count} batches processed, Avg Loss: {avg_epoch_loss:.6f}")
    return model, best_loss, False

def evaluate(dataloader, model, loss_fn, device, writer):
    print()
    print("--- Eval Model ---")
    test_loss = 0.0
    total = 0

    # Get label mappings for display
    dataset = dataloader.dataset
    quality_idx_to_name = {idx: name for name, idx in dataset.quality_label_map.items()}
    type_idx_to_name = {idx: name for name, idx in dataset.section_label_map.items()}

    # Confidence tracking
    quality_confidences = []
    type_confidences = []
    low_confidence_quality = 0
    low_confidence_type = 0
    CONFIDENCE_THRESHOLD = 0.7

    model.eval()

    # Initialize torchmetrics
    # Quality metrics
    quality_accuracy = Accuracy(task="multiclass", num_classes=num_quality_classes, average='macro').to(device)
    quality_precision = Precision(task="multiclass", num_classes=num_quality_classes, average='macro').to(device)
    quality_recall = Recall(task="multiclass", num_classes=num_quality_classes, average='macro').to(device)
    quality_f1 = F1Score(task="multiclass", num_classes=num_quality_classes, average='macro').to(device)

    # Type metrics
    type_accuracy = Accuracy(task="multiclass", num_classes=num_type_classes, average='macro').to(device)
    type_precision = Precision(task="multiclass", num_classes=num_type_classes, average='macro').to(device)
    type_recall = Recall(task="multiclass", num_classes=num_type_classes, average='macro').to(device)
    type_f1 = F1Score(task="multiclass", num_classes=num_type_classes, average='macro').to(device)

    # Per-class metrics
    quality_precision_per_class = Precision(task="multiclass", num_classes=num_quality_classes, average=None).to(device)
    quality_recall_per_class = Recall(task="multiclass", num_classes=num_quality_classes, average=None).to(device)
    type_precision_per_class = Precision(task="multiclass", num_classes=num_type_classes, average=None).to(device)
    type_recall_per_class = Recall(task="multiclass", num_classes=num_type_classes, average=None).to(device)

    with torch.no_grad():
        batch_count = 0
        for batch, (x, quality_label, type_label) in enumerate(dataloader):
            x, quality_label, type_label = x.to(device), quality_label.to(device), type_label.to(device)

            # Use the new prediction method with confidence scores
            predictions = model.predict_with_confidence(x)

            q_preds = predictions['quality_preds']
            q_confidence = predictions['quality_confidence']
            t_preds = predictions['type_preds']
            t_confidence = predictions['type_confidence']

            batch_size = quality_label.size(0)
            total += batch_size
            batch_count += 1

            # Update torchmetrics
            quality_accuracy.update(q_preds, quality_label)
            quality_precision.update(q_preds, quality_label)
            quality_recall.update(q_preds, quality_label)
            quality_f1.update(q_preds, quality_label)

            type_accuracy.update(t_preds, type_label)
            type_precision.update(t_preds, type_label)
            type_recall.update(t_preds, type_label)
            type_f1.update(t_preds, type_label)

            quality_precision_per_class.update(q_preds, quality_label)
            quality_recall_per_class.update(q_preds, quality_label)
            type_precision_per_class.update(t_preds, type_label)
            type_recall_per_class.update(t_preds, type_label)

            # Track confidence scores
            quality_confidences.extend(q_confidence.cpu().numpy())
            type_confidences.extend(t_confidence.cpu().numpy())

            # Count low confidence predictions
            low_confidence_quality += (q_confidence < CONFIDENCE_THRESHOLD).sum().item()
            low_confidence_type += (t_confidence < CONFIDENCE_THRESHOLD).sum().item()

            # Get logits for loss calculation
            pred_quality, pred_type = model(x)
            loss_q = loss_fn(pred_quality, quality_label)
            loss_t = loss_fn(pred_type, type_label)
            test_loss += (loss_q + loss_t)
            print("Batch {}: Loss = {:.6f} (Quality {:.6f}, Type {:.6f})".format(batch, loss_q + loss_t, loss_q, loss_t))

    writer.add_scalar("Loss/test", test_loss / total)

    # Compute final metrics
    import numpy as np

    # Get final metric values
    quality_acc_val = quality_accuracy.compute().item()
    quality_prec_val = quality_precision.compute().item()
    quality_rec_val = quality_recall.compute().item()
    quality_f1_val = quality_f1.compute().item()

    type_acc_val = type_accuracy.compute().item()
    type_prec_val = type_precision.compute().item()
    type_rec_val = type_recall.compute().item()
    type_f1_val = type_f1.compute().item()

    # Per-class metrics
    quality_prec_per_class = quality_precision_per_class.compute().cpu().numpy()
    quality_rec_per_class = quality_recall_per_class.compute().cpu().numpy()
    type_prec_per_class = type_precision_per_class.compute().cpu().numpy()
    type_rec_per_class = type_recall_per_class.compute().cpu().numpy()

    # Calculate overall accuracy (not macro-averaged)
    correct_quality = 0
    correct_type = 0
    correct_both = 0

    # Reset dataloader to calculate overall accuracy
    dataloader_iter = iter(dataloader)
    for batch in dataloader_iter:
        x, quality_label, type_label = batch
        x, quality_label, type_label = x.to(device), quality_label.to(device), type_label.to(device)
        predictions = model.predict_with_confidence(x)
        q_preds = predictions['quality_preds']
        t_preds = predictions['type_preds']

        correct_quality += (q_preds == quality_label).sum().item()
        correct_type += (t_preds == type_label).sum().item()
        correct_both += ((q_preds == quality_label) & (t_preds == type_label)).sum().item()

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print("Using metrics library: torchmetrics")
    print("Total Samples: ", total)
    print("\n--- Accuracy Metrics ---")
    print("Correct Quality Predictions: ", correct_quality)
    print("Correct Type Predictions: ", correct_type)
    print("Correct Both (quality + type): ", correct_both)
    print(f"Test Loss: {test_loss / max(total,1):.4f}")
    print(f"Quality Accuracy = {int(100 * correct_quality / max(total,1))}%")
    print(f"Type Accuracy = {int(100 * correct_type / max(total,1))}%")
    print(f"Combined Accuracy = {int(100 * correct_both / max(total,1))}%")

    # Log accuracy metrics
    logger.info(f"Evaluation completed - Total samples: {total}")
    logger.info(f"Quality Accuracy: {correct_quality}/{total} ({100 * correct_quality / max(total,1):.2f}%)")
    logger.info(f"Type Accuracy: {correct_type}/{total} ({100 * correct_type / max(total,1):.2f}%)")
    logger.info(f"Combined Accuracy: {correct_both}/{total} ({100 * correct_both / max(total,1):.2f}%)")
    logger.info(f"Test Loss: {test_loss / max(total,1):.4f}")

    print("\n--- Precision, Recall & F1 Metrics (Macro Average) ---")
    print(f"Quality Precision: {quality_prec_val:.4f}")
    print(f"Quality Recall: {quality_rec_val:.4f}")
    print(f"Quality F1-Score: {quality_f1_val:.4f}")
    print(f"Type Precision: {type_prec_val:.4f}")
    print(f"Type Recall: {type_rec_val:.4f}")
    print(f"Type F1-Score: {type_f1_val:.4f}")

    # Log precision, recall, and F1 metrics
    logger.info(f"Quality Precision: {quality_prec_val:.4f}")
    logger.info(f"Quality Recall: {quality_rec_val:.4f}")
    logger.info(f"Quality F1-Score: {quality_f1_val:.4f}")
    logger.info(f"Type Precision: {type_prec_val:.4f}")
    logger.info(f"Type Recall: {type_rec_val:.4f}")
    logger.info(f"Type F1-Score: {type_f1_val:.4f}")

    print("\n--- Per-Class Precision & Recall ---")
    print("Quality Classes:")
    for i, (prec, rec) in enumerate(zip(quality_prec_per_class, quality_rec_per_class)):
        class_name = quality_idx_to_name.get(i, f"Class {i}")
        print(f"  {class_name}: Precision = {prec:.4f}, Recall = {rec:.4f}")
        logger.info(f"Quality Class '{class_name}' (idx {i}): Precision = {prec:.4f}, Recall = {rec:.4f}")
    print("Type Classes:")
    for i, (prec, rec) in enumerate(zip(type_prec_per_class, type_rec_per_class)):
        class_name = type_idx_to_name.get(i, f"Class {i}")
        print(f"  {class_name}: Precision = {prec:.4f}, Recall = {rec:.4f}")
        logger.info(f"Type Class '{class_name}' (idx {i}): Precision = {prec:.4f}, Recall = {rec:.4f}")

    print("\n--- Confidence Score Metrics ---")
    print(f"Average Quality Confidence: {np.mean(quality_confidences):.4f}")
    print(f"Average Type Confidence: {np.mean(type_confidences):.4f}")
    print(f"Quality predictions below threshold ({CONFIDENCE_THRESHOLD}): {low_confidence_quality}/{total}")
    print(f"Type predictions below threshold ({CONFIDENCE_THRESHOLD}): {low_confidence_type}/{total}")
    print("="*80 + "\n")

    # Log confidence metrics
    logger.info(f"Average Quality Confidence: {np.mean(quality_confidences):.4f}")
    logger.info(f"Average Type Confidence: {np.mean(type_confidences):.4f}")
    logger.info(f"Quality predictions below confidence threshold ({CONFIDENCE_THRESHOLD}): {low_confidence_quality}/{total}")
    logger.info(f"Type predictions below confidence threshold ({CONFIDENCE_THRESHOLD}): {low_confidence_type}/{total}")
    logger.info("Evaluation completed successfully")

    return test_loss / batch_count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", device)
    logger.info(f"Training started on device: {device}")

    print()
    print("--- Tensorboard Setup---")
    writer = SummaryWriter(LOG_DIR)

    print()
    print("--- Instantiate Model ---")
    model = EstateInsightModel(num_quality_classes=3, num_type_classes=5).to(device)
    # model = PreTrainedModel().to(device)
    best_loss = float('inf')
    

    NUM_EPOCHS = 20
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001
    )
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stop = EarlyStop() # patience = 10


    print("--- Load Best Model ---")
    if os.path.exists(MODEL_PATH):
        best_model = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(best_model['model_state_dict'])
        optimizer.load_state_dict(best_model['optimizer_state_dict'])
        best_loss = best_model['loss']
        print("Loaded best model from ", MODEL_PATH)
        logger.info(f"Loaded existing model from {MODEL_PATH} with best loss: {best_loss:.6f}")
    else:
        logger.info("No existing model found, starting training from scratch")

    logger.info(f"Starting training for {NUM_EPOCHS} epochs")

    for epoch in range(NUM_EPOCHS):
        model, best_loss, should_stop = train(train_loader, model, criterion, best_loss, optimizer, epoch, early_stop, device, writer)
        test_loss = evaluate(test_loader, model, criterion, device, writer)
        scheduler.step()

        if should_stop:
            print(f"Model has stopped training after {epoch+1}")
            logger.info(f"Training stopped early after {epoch+1} epochs due to early stopping")
            break

    writer.close()
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()