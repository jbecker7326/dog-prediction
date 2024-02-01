import tflite_runtime.interpreter as tflite
import os
import numpy as np
import streamlit as st

from PIL import Image


def batch_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img = img.resize((299, 299), Image.NEAREST)    
    X = np.array(img, dtype='float32')
    batch = np.expand_dims(X, axis=0)

    return batch


def predict(image):

    classes = ['Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih tzu', 'Blenheim spaniel', 'Papillon', 'Toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'Basset', 'Beagle', 'Bloodhound', 'Bluetick', 'Black and tan coonhound', 'Walker hound', 'English foxhound', 'Redbone', 'Borzoi', 'Irish wolfhound', 'Italian greyhound', 'Whippet', 'Ibizan hound', 'Norwegian elkhound', 'Otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier', 'American staffordshire terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'Wire haired fox terrier', 'Lakeland terrier', 'Sealyham terrier', 'Airedale', 'Cairn', 'Australian terrier', 'Dandie dinmont', 'Boston bull', 'Miniature schnauzer', 'Giant schnauzer', 'Standard schnauzer', 'Scotch terrier', 'Tibetan terrier', 'Silky terrier', 'Soft coated wheaten terrier', 'West highland white terrier', 'Lhasa', 'Flat coated retriever', 'Curly coated retriever', 'Golden retriever', 'Labrador retriever', 'Chesapeake bay retriever', 'German short haired pointer', 'Vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel', 'Clumber', 'English springer', 'Welsh springer spaniel', 'Cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'Kuvasz', 'Schipperke', 'Groenendael', 'Malinois', 'Briard', 'Kelpie', 'Komondor', 'Old english sheepdog', 'Shetland sheepdog', 'Collie', 'Border collie', 'Bouvier des flandres', 'Rottweiler', 'German shepherd', 'Doberman', 'Miniature pinscher', 'Greater swiss mountain dog', 'Bernese mountain dog', 'Appenzeller', 'Entlebucher', 'Boxer', 'Bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great dane', 'Saint bernard', 'Eskimo dog', 'Malamute', 'Siberian husky', 'Affenpinscher', 'Basenji', 'Pug', 'Leonberg', 'Newfoundland', 'Great pyrenees', 'Samoyed', 'Pomeranian', 'Chow', 'Keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'Toy poodle', 'Miniature poodle', 'Standard poodle', 'Mexican hairless', 'Dingo', 'Dhole', 'African hunting dog']

    interpreter = tflite.Interpreter(model_path=os.path.joint('..', 'models', 'model.tflite'))
    interpreter.allocate_tensors()
    
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    
    batch = batch_image(img)

    interpreter.set_tensor(input_index, batch)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()
    dict_predictions = dict(zip(classes, float_predictions))
    top_5 = sorted(dict_predictions, key=dict_predictions.get, reverse=True)[:5]
    #prediction = classes[np.argmax(float_predictions)]

    return top_5


def main():

    st.title("Dog Breed Classification")
    image_file = st.file_uploader(
        "Upload Image for classification", type=["jpg", "jpeg", "heic"]
    )
    if image_file is not None:
        img = Image.open(image_file)
        st.text("Processing image...")
        st.image(img)

    if st.button("Predict"):
        result = predict(img)
        st.text("Top 5 dog breeds matched with your image:")
        st.write(result)

if __name__ == "__main__":
    main()