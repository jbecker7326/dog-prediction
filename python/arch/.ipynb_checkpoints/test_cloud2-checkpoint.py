import requests
from io import BytesIO
from urllib import request
from PIL import Image
import base64

if __name__ == "__main__":

    #post_url = "https://4bfnidjam6.execute-api.us-east-1.amazonaws.com/deploy-1/predict"
    post_url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
   
    image_url = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Redbone-coonhound-detail.jpg'
    
    with request.urlopen(image_url) as resp:
        buffer = resp.read()
        
    stream = BytesIO(buffer)
    img = Image.open(stream)
    #img = batch_image(img)
        
    bytes=img.tobytes()
    decoded_image = base64.b64encode(bytes).decode("utf8")

    data = {'image': decoded_image, 'size': img.size}

    result = requests.post(post_url, json=data).json()
    print(result)