import sagemaker
import torch
import torch.nn as nn
import torch.optim as optim
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.serializers import JSONSerializer, DataSerializer
from sagemaker.deserializers import JSONDeserializer
from src.EstateInsightModel import main, LOG_DIR, MODEL_PATH
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import shutil
import os
import tarfile
import boto3
import io
from dotenv import load_dotenv
import argparse

USE_GPU = True
TRAIN_DEVICE = 'ml.g4dn.xlarge' if USE_GPU else 'ml.m5.large'
DEPLOY_DEVICE = 'ml.m5.large'
LOCAL_MODEL_DIR = 'model'
TAR_NAME = 'model.tar.gz'
NUM_EPOCHS = 2
print(f"training on {TRAIN_DEVICE}")
print(f"deploying on {DEPLOY_DEVICE}")

load_dotenv()

# add a way to accept hyperparameters to arguments
parser = argparse.ArgumentParser()

# sagemaker specific args
parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

# define the hyperparameters
parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
parser.add_argument('--learning_rate', type=float, default=0.001)
args, _ = parser.parse_known_args()

main(args)

code_dir = os.path.join(LOCAL_MODEL_DIR, 'code')
os.makedirs(code_dir, exist_ok=True)

if os.path.exists('src/inference.py'):
    shutil.copy('src/inference.py', os.path.join(code_dir, 'inference.py'))
    #shutil.copy('requirements.txt', os.path.join(code_dir, 'requirements.txt'))
    
# copy estate_insight.pth into the model folder
if os.path.exists(MODEL_PATH):
  shutil.copy(MODEL_PATH, os.path.join(LOCAL_MODEL_DIR, MODEL_PATH))
    
with tarfile.open(TAR_NAME, 'w:gz') as tar:
  tar.add(MODEL_PATH, arcname='estate_insight.pth')
  tar.add(code_dir, arcname='code')

  
print(f"Saved model to {TAR_NAME}")

# iam_client = boto3.client('iam')
# role_response = iam_client.get_role(RoleName='AmazonSageMaker-ExecutionRole-20260409T093822')
# ARN = role_response['Role']['Arn']

#session = sagemaker.Session(boto3.Session(region_name='us-east-1'))
ARN = os.getenv('ARN')
REGION = os.getenv('AWS_REGION')
os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'
try:
  session = sagemaker.Session(boto3.Session(region_name='us-east-1'))
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
  framework_version='2.1.0',
  py_version='py310',
  entry_point='inference.py',
  sagemaker_session=session,
  source_dir='model/code/',
  env={
        'TS_MAX_RESPONSE_WAIT_TIME': '300',      # TorchServe timeout (5 mins)
        'SAGEMAKER_MODEL_SERVER_TIMEOUT': '300', # SageMaker container timeout
        'MODEL_SERVER_TIMEOUT': '300',
        'SAGEMAKER_MODEL_SERVER_WORKERS': '1'
    }
)

predictor = pytorch_model.deploy(
  initial_instance_count=1,
  instance_type=TRAIN_DEVICE,
  serializer=DataSerializer(content_type="image/jpeg"),
  deserializer=JSONDeserializer()
)

# Find a test image and send it to same directory as this script, name it 'img.jpg'
# img_pth = 'img.jpg'
# image = Image.open(img_pth).convert('RGB')
# # can't send PIL images to our model in the cloud, must send bytes instead
# buffer = io.BytesIO()
# # You must specify a format (e.g., JPEG or PNG) to compress the object into bytes
# image.save(buffer, format="JPEG") 
# payload = buffer.getvalue()
# response = predictor.predict(payload)
# train_dataset.quality_label_map
# print()
# print(response)
# print(f"Endpoint: {predictor.endpoint_name}")
