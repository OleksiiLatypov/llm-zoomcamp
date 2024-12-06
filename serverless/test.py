# import requests

# # URL of the Flask app's /predict endpoint
# url = "http://localhost:9696/predict"

# # Image path to test
# image_path = "/workspaces/llm-zoomcamp/serverless/data/test/curly/1a9dbe23a0d95f1c292625960e4509184.jpg"  # Replace with the actual image path

# # Prepare the payload with the image path
# data = {"image_path": image_path}

# # Send a POST request with the image path as JSON
# response = requests.post(url, json=data)

# # Parse the JSON response
# response_data = response.json()

# # Print the response (prediction)
# print(response_data)

import requests

# URL of the Flask app's /predict endpoint
url = "http://localhost:9696/predict"

# Image path to test
image_path = "/workspaces/llm-zoomcamp/serverless/data/train/curly/image21.jpeg"  # Replace with the actual image path

# Open the image file in binary mode and send it in a POST request
with open(image_path, 'rb') as image_file:
    files = {'file': image_file}  # Flask expects a file field called 'file'
    
    # Send the POST request with the image file
    response = requests.post(url, files=files)

# Check if the response status code is in the 2xx range (success)
if response.status_code == 200:
    # If the request was successful, parse the JSON response
    response_data = response.json()
    print("Prediction response:", response_data)
else:
    # If the request failed, print the status code and message
    print(f"Request failed with status code {response.status_code}. Response: {response.text}")

