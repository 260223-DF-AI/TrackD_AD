from PIL import Image
from torchvision import transforms
from collections import OrderedDict
import torch.nn as nn
import torchvision.models as models
import torch
import json
import os
import io

class EstateInsightModel(nn.Module):
    def __init__(self, num_quality_classes, num_type_classes):
        super(EstateInsightModel, self).__init__()
        # Load a pre-trained ResNet model
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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
            if "layer4" in name or "fc" in name or 'layer3' in name:
                param.requires_grad = True

    def forward(self, x):
        features = self.model(x)
        quality_output = self.quality_head(features)
        type_output = self.type_head(features)
        return quality_output, type_output

def model_fn(model_dir):
  print("---1. Entering model_fn---")
  try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"---2. Device: {device}---")
    
    num_quality_classes, num_type_classes = 3, 5
    model = EstateInsightModel(num_quality_classes, num_type_classes)
    print(f"---3. Class Initialized---")
    
    model_path = os.path.join(model_dir, 'estate_insight.pth')
    checkpoint = torch.load(model_path, map_location=device)
    print("---4. Checkpoint Loaded---")

    # Load the weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Attach the labels to the model object so predict_fn can see them
    model.quality_label_names = checkpoint['quality_label_names']
    model.house_section_label_names = checkpoint['house_section_label_names']

    model.to(device)
    model.eval()
    return model
    
  except Exception as e:
    print(f"FAILED TO LOAD MODEL {str(e)}") 
    raise e

def input_fn(request_body, request_content_type):
  
  # original code that doesn't support PIL images:
  if request_content_type == 'application/json':
    body = request_body.decode('utf-8') if isinstance(request_body, bytes) else request_body
    data = json.loads(body)
    return torch.tensor(data, dtype=torch.float32).view(-1, 1)
  
  if request_content_type == 'image/jpeg':
    image = Image.open(io.BytesIO(request_body)).convert("RGB")
    preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)
    
  raise ValueError(f"Unsupported format: {request_content_type}")

def predict_fn(input_data, model):
  try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_data = input_data.to(device)
    
    model.eval()
    with torch.no_grad():
      quality_pred, section_pred = model(input_data.to(device))

      # convert the predictions to probabilities usign softmax, then move to cpu
      quality_probability = torch.softmax(quality_pred, dim=1).cpu().numpy().flatten()
      section_probability = torch.softmax(section_pred, dim=1).cpu().numpy().flatten()

    # convert probabilities to floats since Sagemaker JSON serializers crash on numpy / tensor values
    return {
        "all_quality_scores": {
            label: float(prob) for label, prob in zip(model.quality_label_names, quality_probability)
        },
        "all_section_scores": {
            label: float(prob) for label, prob in zip(model.house_section_label_names, section_probability)
        },
        "prediction": {
            "quality": str(model.quality_label_names[quality_probability.argmax()]),
            "section": str(model.house_section_label_names[section_probability.argmax()])
        }
    }
  except Exception as e:
    print(f"Exception occured: {e}")
    raise e

def output_fn(prediction, accept):
  if accept == 'application/json':
    return json.dumps(prediction), accept
  raise ValueError(f"unsupported accept type: {accept}")
  