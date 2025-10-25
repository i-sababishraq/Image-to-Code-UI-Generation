import json
import os
from openai import AzureOpenAI
import base64


endpoint = os.getenv("ENDPOINT_URL", "https://gaea-testing.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "BmeB9EiGuHu4mgNIuICTqjsqxv9j3dSvHKN2BUxXlq4etz3Te1xTJQQJ99BAACHYHv6XJ3w3AAABACOG7Kq1")

# import os
from openai import OpenAI

os.environ["AZURE_OPENAI_API_KEY"] = "BmeB9EiGuHu4mgNIuICTqjsqxv9j3dSvHKN2BUxXlq4etz3Te1xTJQQJ99BAACHYHv6XJ3w3AAABACOG7Kq1"

client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url="https://gaea-testing.openai.azure.com/openai/v1/",
)

response = client.responses.create(   
  model="gpt-4.1-mini", # Replace with your model deployment name 
  input="This is a test.",
)

print(response.model_dump_json(indent=2))