import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
headers = {"Authorization": f"Bearer {api_key}"}

try:
    response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
    if response.status_code == 200:
        models = response.json().get("data", [])
        print("Available Groq Models:")
        for model in sorted(models, key=lambda x: x["id"]):
            print(f"- {model['id']}")
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"Exception: {e}")
