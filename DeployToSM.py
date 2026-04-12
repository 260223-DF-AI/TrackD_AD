import sagemaker
import torch
import torch.nn as nn
import torch.optim as optim
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.serializers import JSONSerializer, DataSerializer
from sagemaker.deserializers import JSONDeserializer
from src.EstateInsightModel import EstateInsightModel, EarlyStop, main, LOG_DIR, MODEL_PATH
from src.EstateInsightModel import train
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import shutil
import os
import tarfile
import boto3
import io

USE_GPU = True
TRAIN_DEVICE = 'ml.g4dn.xlarge' if USE_GPU else 'ml.m5.large'
DEPLOY_DEVICE = 'ml.m5.large'
LOCAL_MODEL_DIR = 'model'
TAR_NAME = 'model.tar.gz'
NUM_EPOCHS = 1
print(f"training on {TRAIN_DEVICE}")
print(f"deploying on {DEPLOY_DEVICE}")


# main()

code_dir = os.path.join(LOCAL_MODEL_DIR, 'code')
if os.path.exists('src/inference.py'):
    shutil.copy('src/inference.py', os.path.join(code_dir, 'inference.py'))
    shutil.copy('requirements.txt', os.path.join(code_dir, 'requirements.txt'))
    
with tarfile.open(TAR_NAME, 'w:gz') as tar:
  tar.add(MODEL_PATH, arcname='estate_insight.pth')
  tar.add(code_dir, arcname='code')

  
print(f"Saved model to {TAR_NAME}")

iam_client = boto3.client('iam')
role_response = iam_client.get_role(RoleName='AmazonSageMaker-ExecutionRole-20260409T093822')
ARN = role_response['Role']['Arn']

try:
  session = sagemaker.Session()
  try:
    role = sagemaker.get_execution_role()
  except (ValueError, RuntimeError):
    role = ARN
    
  bucket = session.default_bucket()
  print(f"Bucket: {bucket}")
  
except Exception as e:
  print(e)
  exit(1)

s3_prefix = 'EstateInsightDemo'
s3_model_path = session.upload_data(path=TAR_NAME, bucket=bucket, key_prefix=s3_prefix)
print(f"Uploaded model to {s3_model_path}")

pytorch_model = PyTorchModel(
  model_data=s3_model_path,
  role=role,
  framework_version='2.0.0',
  py_version='py310',
  entry_point='inference.py',
  sagemaker_session=session
)

predictor = pytorch_model.deploy(
  initial_instance_count=1,
  instance_type=DEPLOY_DEVICE,
  serializer=DataSerializer(content_type="image/jpeg"),
  deserializer=JSONDeserializer()
)

# Find a test image and send it to same directory as this script, name it 'img.jpg'
img_pth = 'img.jpg'
image = Image.open(img_pth).convert('RGB')
buffer = io.BytesIO()
# You must specify a format (e.g., JPEG or PNG) to compress the object into bytes
image.save(buffer, format="JPEG") 
payload = buffer.getvalue()
response = predictor.predict(payload)
print(response)
