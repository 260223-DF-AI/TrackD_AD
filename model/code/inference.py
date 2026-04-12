from .EstateInsightModel import EstateInsightModel
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch
import json
import os
import io

def model_fn(model_dir):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = EstateInsightModel()
  model_path = os.path.join(model_dir, 'estate_insight.pth')
  with(open(model_path, 'rb') as f):
    model.load_state_dict(torch.load(f, map_location=device))
  
  return model.to(device)

def input_fn(request_body, request_content_type):
  
  # original code that doesn't support PIL images:
  if request_content_type == 'application/json':
    body = request_body.decode('utf-8') if isinstance(request_body, bytes) else request_body
    data = json.loads(body)
    return torch.tensor(data, dtype=torch.float32).view(-1, 1)
  
  if request_content_type == 'image/jpeg':
    image = Image.open(io.BytesIO(request_body).convert("RGB"))
    preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #return preprocess(image).unsqueeze(0)
    return preprocess(image)
  
  raise ValueError(f"Unsupported format: {request_content_type}")

def predict_fn(input_data, model):
  try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_data = input_data.to(device)
    
    model.eval()
    with torch.no_grad():
      prediction = model(input_data)
    return prediction.cpu().numpy()
  
  except Exception as e:
    print(f"Exception occured: {e}")
    raise e

def output_fn(prediction, content_type):
  return json.dumps(prediction.tolist())
  