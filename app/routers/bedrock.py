from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
#from openai import OpenAI
from dotenv import load_dotenv
from matplotlib import image
from pydantic import BaseModel
from DeployedModel import get_endpoint, sagemaker_prediction
from anthropic import AnthropicBedrock
import os
import boto3
import json
import base64

class Prompt(BaseModel):
  prompt: str
  max_tokens: int = 1000

router = APIRouter()
load_dotenv()

client = AnthropicBedrock(
  api_key=os.environ.get('AWS_BEARER_TOKEN_BEDROCK'),
  aws_region='us-east-1'
)

@router.post("")
async def SendPrompt(image: UploadFile = File(...)):
  
  image_bytes = await image.read()
  image_base64 = base64.b64encode(image_bytes).decode("utf-8")
  
  endpoint = get_endpoint()
  predictions = sagemaker_prediction(image_bytes, endpoint)
  print(type(predictions))
  prompt = "If the input image isn't an image of a house or a room inside a house, respond with 'I am sorry, but I can only analyze images of houses or rooms' and end your response here. Do not offer to use your own insights without a valid house image with model predictions."
  prompt += f"The following image has been analyzed and the model predicts the following features:  + {json.dumps(predictions)} + . "
  prompt += "Start off by stating what the predicted labels of the image were. E.g. 'The model predicted this image was a bathroom with a probability of X%, with a basic quality score of Y%'"
  prompt += "If the model's predictions are None, or invalid. State the model wasn't able to make a prediction and end your response here. Do not offer to use your own insights without the model predictions."
  prompt += "Format the output in sections. For example, if talking about the fixtures, format that part of the response like this: ***Fixtures*** - Your reasoning here, or ***Sales Pitch*** - your reasoning here."
  prompt += "Make sure that the reasoning you give mentions why the features predicted by the model are accurate, and note the quality the model predicted e.g. old, basic, renovated, are properly attributed to the image."
  prompt += "Give me a concise analysis of the image supporting the model's predictions, and then give pricing strategies and sales pitch for the property based on the model's predictions if this room/section were a part of a house to be added to a real estate listing."
  prompt += "Go through your response step by step and make sure to support your reasoning with the model's predictions. If the model's predictions are of low quality, make sure to note that in your analysis and reasoning."
  
  print("Generating Response...")
  response = client.messages.create(
    # 'Inference profile arn' needs to be used
    model='us.anthropic.claude-sonnet-4-20250514-v1:0',
    max_tokens=1000,
    messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.content_type,
                        "data": image_base64,
                    },
                },
                {"type": "text", "text": prompt}
            ],
        }],
  )
  
  return {'analysis:': response.content[0].text}

@router.post("/TreeOfThoughtPrompt")
async def TreeOfThoughtPrompt(image: UploadFile = File(...)):
  image_bytes = await image.read()
  image_base64 = base64.b64encode(image_bytes).decode("utf-8")
  
  endpoint = get_endpoint()
  predictions = sagemaker_prediction(image_bytes, endpoint)
  print(type(predictions))
  prompt = "If the input image isn't an image of a house or a room inside a house, respond with 'I am sorry, but I can only analyze images of houses or rooms' and end your response here. Do not offer to use your own insights without a valid house image with model predictions."
  prompt += f"The following image has been analyzed and the model predicts the following features:  + {json.dumps(predictions)} + . "
  
  # Tree of thought prompt text file path
  tot_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'prompts', 'ToT.txt'))
  try:
      with open(tot_path, 'r', encoding='utf-8') as f:
          prompt += "\n" + f.read()
  except FileNotFoundError:
      prompt += "\n" + "ToT.txt not found."

  print("Generating Response...")
  response = client.messages.create(
    # 'Inference profile arn' needs to be used
    model='us.anthropic.claude-sonnet-4-20250514-v1:0',
    max_tokens=1000,
    messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.content_type,
                        "data": image_base64,
                    },
                },
                {"type": "text", "text": prompt}
            ],
        }],
  )
  
  return {'analysis:': response.content[0].text}