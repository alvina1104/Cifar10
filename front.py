import streamlit as st
import requests
from PIL import Image

st.title('CIFAR-10 PREDICTION')
st.header('Upload an image')

api_url = 'http://127.0.0.1:8000/predict/'

uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded image', use_container_width=True)

    if st.button('Predict'):
        files = {'file': uploaded_file.getvalue()}

        try:
            response = requests.post(api_url, files=files)

            if response.status_code == 200:
                result = response.json()

                st.success(f"The image is {result['label']}")

            else:
                st.error("The image was not uploaded")

        except Exception as e:
            st.error(f'Server is unavailable: {e}')