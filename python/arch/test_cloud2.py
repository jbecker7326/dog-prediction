import requests
from io import BytesIO
from urllib import request
from PIL import Image
import base64
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-t", "--test", help="test type local or cloud")
args = parser.parse_args()

def main():
    print(args)
    post_url = "https://4bfnidjam6.execute-api.us-east-1.amazonaws.com/deploy-1/predict"
   
    image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Redbone-coonhound-detail.jpg'
    
    with request.urlopen(image_url) as resp:
        buffer = resp.read()
        
    stream = BytesIO(buffer)
    img = Image.open(stream)
        
    bytes=img.tobytes()
    decoded_image = base64.b64encode(bytes).decode("utf8")

    data = {'image': decoded_image, 'size': img.size}

    result = requests.post(post_url, json=data).json()
    print(result)
    
if __name__ == "__main__":
    main()