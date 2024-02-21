import requests
import base64
from io import BytesIO
from PIL import Image
import streamlit as st


def decode_image(img):
    img = img.resize((299, 299), Image.NEAREST)
    bytes=img.tobytes()
    decoded_image = base64.b64encode(bytes).decode("utf8")
    return decoded_image


def main():
    st.title("Dog Breed Classification")
    image_file = st.file_uploader(
        "Upload Image for classification", type=["jpg", "jpeg", "heic"]
    )

    url = "https://4bfnidjam6.execute-api.us-east-1.amazonaws.com/deploy-1/predict"
    #url = 'http://ada480397f770408e91ddfe53fbce180-1951642851.us-east-1.elb.amazonaws.com/predict'

    if image_file is not None:
        with st.spinner("Processing image..."):
          img = Image.open(image_file)
        st.image(img)
        #st.text("Processing image...")
    
        if st.button("Predict"):
            img = decode_image(img)
            data = {'image': img, 'size': (299, 299)}

            with st.spinner("Predicting..."):
                result = requests.post(url, json=data).json()
            
            st.markdown("Match found! :balloon: \n\n **Top 5 dog breeds matched with your image:**")
            
            table = ["Rank | Breed | Score",  "--- | --- | ---"]
            for i in range(len(result)):
                table.append(f"{i+1} | {result[i][0]} | {result[i][1]:.5f}")
            st.markdown("\n".join(table))
            

if __name__ == "__main__":
    main()
    