import json
import os
import re
import pandas as pd
import concurrent.futures
from openai import AzureOpenAI
import base64


endpoint = os.getenv("ENDPOINT_URL", "https://gaea-testing.openai.azure.com/")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-2")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "BmeB9EiGuHu4mgNIuICTqjsqxv9j3dSvHKN2BUxXlq4etz3Te1xTJQQJ99BAACHYHv6XJ3w3AAABACOG7Kq1")  


# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-08-01-preview",
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


json_path = "/home/ro834336/Custom_Bench/gpt4-Bench.json"
with open(json_path, 'r') as f:
    results = json.load(f)


batch = []
for index, result in enumerate(results):
    file_name = result['id']
    image = result['image']
    image_path = f"/home/c3-0/datasets/MP-16/resources/images/mp16/{image}"

    question = result['conversations'][0]['value'].replace('<image>\n','')
    ground_truth = result['conversations'][1]['value']
    
    question_type = result['question_type']

    if question_type == "MCQ":
        prompt_eval = "For the given Multiple Choice Question, analyze the question and answer strictly from one of the options below. Strictly answer the choice only. No additional text.\n" + question

    elif question_type == "TF":
        prompt_eval = "\nThe above question is a True/False question. " + question

    elif question_type == "LVQA":
        prompt_eval = "Answer the question in detail. " + question
    
    elif question_type == "SVQA":
        prompt_eval = "\nPlease provide brief, clear response to the asked question.\n" + question

    base64_image = encode_image(image_path)

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_eval
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        },
        {
            "role": "system",
            "content": "You are a helpful Assistant. Provide helpful response to the user's question."
        }
    ]

    bat = {
        "custom_id": f"{file_name}_{index}",
        "method": "POST", 
        "url": "/v1/chat/completions", 
        "body": {
                "model": "gpt-4o-mini", 
                "messages": PROMPT_MESSAGES,
                "max_tokens": 1000
            }
        }
    
    batch.append(bat)

batc_output_path = "/home/ashmal/GeoGPT/GPT_Evaluation/BatchFiles"
model_id = "GPT-4o"
model_id = "checkpoint_1"

with open(f'{batc_output_path}/{model_id}.jsonl', 'w') as f:
    for entry in batch:
        f.write(json.dumps(entry))
        f.write('\n')