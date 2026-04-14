from PIL import Image
import io
import sagemaker
from sagemaker.predictor import Predictor
import boto3
import json
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import DataSerializer


def sagemaker_prediction(input_data, endpoint_name):
  print("Getting prediction...")
  predictor = Predictor(endpoint_name,
            serializer=DataSerializer(content_type="image/jpeg"),
            deserializer=JSONDeserializer()
                        )
  
  response = predictor.predict(input_data)
  print(response)

def get_endpoint():
  print("Finding endpoint...")
  sagemaker_client = boto3.client('sagemaker')
  
  response = sagemaker_client.list_endpoints(
    StatusEquals='InService'
  )
  
  if response['Endpoints']:
    endpoint_name = response['Endpoints'][0]['EndpointName']
    return endpoint_name
  
  raise ValueError("No endpoint found")

def create_image_payload(img_pth : str):
  """ Creates a BytesIO object since JSON can have bytes, but not PIL Images.

  Args:
      img_pth (str): path to the image

  Returns:
      BytesIO: from the io module
  """
  
  print("Converting Image to Bytes for JSON payload...")
  image = Image.open(img_pth).convert('RGB')
  buffer = io.BytesIO()
  # You must specify a format (e.g., JPEG or PNG) to compress the object into bytes
  image.save(buffer, format="JPEG") 
  payload = buffer.getvalue()
  return payload

def main():
  input_data = "img.jpg"
  endpoint_name = get_endpoint()
  payload = create_image_payload(input_data)
  sagemaker_prediction(payload, endpoint_name=endpoint_name)
  

if __name__ == "__main__":
  print("Starting DeployedModel.py...")
  main()