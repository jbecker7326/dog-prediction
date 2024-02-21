import numpy as np
import tensorflow as tf
from tensorflow import keras

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from io import BytesIO
from PIL import Image
from urllib import request
from argparse import ArgumentParser
import requests
import base64

from proto import np_to_protobuf


parser = ArgumentParser()
parser.add_argument("-t", "--test", choices=['local', 'lambda', 'model', 'gateway', 'kube', 'eks'], required=True,help="test type local or cloud")
args = parser.parse_args()

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def batch_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img = img.resize((299, 299), Image.NEAREST)    
    X = np.array(img, dtype='float32')
    batch = np.expand_dims(X, axis=0)

    return batch

#def np_to_protobuf(data):
#    return tf.make_tensor_proto(data, shape=data.shape)

def main():
    classes = ['Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih tzu', 'Blenheim spaniel', 'Papillon', 'Toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'Basset', 'Beagle', 'Bloodhound', 'Bluetick', 'Black and tan coonhound', 'Walker hound', 'English foxhound', 'Redbone', 'Borzoi', 'Irish wolfhound', 'Italian greyhound', 'Whippet', 'Ibizan hound', 'Norwegian elkhound', 'Otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier', 'American staffordshire terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'Wire haired fox terrier', 'Lakeland terrier', 'Sealyham terrier', 'Airedale', 'Cairn', 'Australian terrier', 'Dandie dinmont', 'Boston bull', 'Miniature schnauzer', 'Giant schnauzer', 'Standard schnauzer', 'Scotch terrier', 'Tibetan terrier', 'Silky terrier', 'Soft coated wheaten terrier', 'West highland white terrier', 'Lhasa', 'Flat coated retriever', 'Curly coated retriever', 'Golden retriever', 'Labrador retriever', 'Chesapeake bay retriever', 'German short haired pointer', 'Vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel', 'Clumber', 'English springer', 'Welsh springer spaniel', 'Cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'Kuvasz', 'Schipperke', 'Groenendael', 'Malinois', 'Briard', 'Kelpie', 'Komondor', 'Old english sheepdog', 'Shetland sheepdog', 'Collie', 'Border collie', 'Bouvier des flandres', 'Rottweiler', 'German shepherd', 'Doberman', 'Miniature pinscher', 'Greater swiss mountain dog', 'Bernese mountain dog', 'Appenzeller', 'Entlebucher', 'Boxer', 'Bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great dane', 'Saint bernard', 'Eskimo dog', 'Malamute', 'Siberian husky', 'Affenpinscher', 'Basenji', 'Pug', 'Leonberg', 'Newfoundland', 'Great pyrenees', 'Samoyed', 'Pomeranian', 'Chow', 'Keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'Toy poodle', 'Miniature poodle', 'Standard poodle', 'Mexican hairless', 'Dingo', 'Dhole', 'African hunting dog']
    image_url = "https://upload.wikimedia.org/wikipedia/commons/4/4b/Golden_retriever_running_on_a_dirt_road.jpg"

    if args.test == "local":
        url = 'http://localhost:8080/2015-03-31/functions/function/invocations'        
    elif args.test == "lambda":
        url = "https://4bfnidjam6.execute-api.us-east-1.amazonaws.com/deploy-1/predict"    
    elif args.test == "model":
        url = 'localhost:8500'

        channel = grpc.insecure_channel(url)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        pb_request = predict_pb2.PredictRequest()
        pb_request.model_spec.name = 'converted_model'
        pb_request.model_spec.signature_name = 'serving_default'

        img = download_image(image_url)
        X = batch_image(img)
        pb_request.inputs['input_60'].CopyFrom(np_to_protobuf(X))
        
        pb_response = stub.Predict(pb_request, timeout=20.0)
        preds = pb_response.outputs['dense_113'].float_val

        dict_predictions = dict(zip(classes, preds))
        top_5 = sorted(dict_predictions.items(), key=lambda x:x[1], reverse=True)[:5]
        print(top_5)
        return
    elif args.test == "gateway":
        url = 'http://localhost:9696/predict'
    elif args.test == "kube":
        url = 'http://localhost:8080/predict'
    elif args.test == "eks":
        url = 'http://ac842723604204b1d8b94efb760d38a7-1664883915.us-east-1.elb.amazonaws.com/predict'

    img = download_image(image_url)

    img = img.resize((299, 299), Image.NEAREST)
    bytes=img.tobytes()
    decoded_image = base64.b64encode(bytes).decode("utf8")

    data = {'image': decoded_image, 'size': (299, 299)}
    result = requests.post(url, json=data).json()
    print(result)


if __name__ == "__main__":
    main()