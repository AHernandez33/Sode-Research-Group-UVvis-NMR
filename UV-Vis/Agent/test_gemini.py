import os
from google import genai

client = genai.Client(
    api_key=os.getenv("api_key")
)

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Return exactly: working"
)

print(response.text)