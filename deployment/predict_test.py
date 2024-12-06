import requests

# Step 1: Define the URL of your Flask app
url = "http://localhost:9690/predict"  # or use your public URL

# Step 2: Define the client data

client = {"job": "student", "duration": 280, "poutcome": "failure"}
#client = {"job": "management", "duration": 400, "poutcome": "success"}

try:
    response = requests.post(url, json=client)
    print(response.json())
except requests.exceptions.ConnectionError as e:
    print(f"Connection error: {e}")

#{'probability': 0.756743795240796}