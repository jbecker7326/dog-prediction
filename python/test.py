import requests
import base64
from io import BytesIO
from PIL import Image
from urllib import request
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-t", "--test", choices=['local', 'cloud'], required=True,help="test type local or cloud")
args = parser.parse_args()


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def main():
    if args.test == "local":
        post_url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
    else:
        post_url = "https://4bfnidjam6.execute-api.us-east-1.amazonaws.com/deploy-1/predict"

    image_url = "https://upload.wikimedia.org/wikipedia/commons/4/4b/Golden_retriever_running_on_a_dirt_road.jpg"
    
    img = download_image(image_url)
    
    img = img.resize((299, 299), Image.NEAREST)
    bytes=img.tobytes()
    decoded_image = base64.b64encode(bytes).decode("utf8")

    data = {'image': decoded_image, 'size': (299, 299)}
    result = requests.post(post_url, json=data).json()

    print(result)


if __name__ == "__main__":
    main()