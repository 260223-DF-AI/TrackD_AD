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
   # 3. Pull the specific state_dict key you used during save
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # 4. Clean 'module.' prefixes if you trained with DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # 5. Load with strict=False to ignore minor version/metadata differences
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model
    
  #   if 'model_state_dict' in checkpoint:
  #       model.load_state_dict(checkpoint['model_state_dict'])
  #   else:
  #     model.load_state_dict(checkpoint)
      
  #   model.to(device)
  #   model.eval()
  #   return model
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
      prediction1, prediction2 = model(input_data)
    result1 = [float(x) for x in prediction1.cpu().numpy().flatten()]
    result2 = [float(x) for x in prediction2.cpu().numpy().flatten()]
    #return {"test": [1.0, 2.0], "status": "ok"}
    print(f"quality: {result1}, type: {result2}")
    return {"Quality" : result1, "Type": result2}
  
  except Exception as e:
    print(f"Exception occured: {e}")
    raise e

def output_fn(prediction, accept):
  if accept == 'application/json':
    return json.dumps(prediction), accept
  raise ValueError(f"unsupported accept type: {accept}")
  