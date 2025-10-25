import json
import os
from openai import AzureOpenAI, OpenAI
from langchain.chat_models import init_chat_model
import base64

os.environ["AZURE_OPENAI_API_KEY"] = "BmeB9EiGuHu4mgNIuICTqjsqxv9j3dSvHKN2BUxXlq4etz3Te1xTJQQJ99BAACHYHv6XJ3w3AAABACOG7Kq1"

client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url="https://gaea-testing.openai.azure.com/openai/v1/",
)

# response = client.responses.create(   
#   model="gpt-4.1-mini",
#   input="This is a test.",
# )

# print(response.model_dump_json(indent=2))






os.environ["OPENAI_API_KEY"] = "sk-..."

llm = init_chat_model("openai:gpt-4.1")