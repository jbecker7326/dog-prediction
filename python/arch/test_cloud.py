import requests

url = "https://hciuk807s3.execute-api.us-east-2.amazonaws.com/test/predict"

data = {'url': 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Redbone-coonhound-detail.jpg'}

result = requests.post(url, json=data).json()
print(result)