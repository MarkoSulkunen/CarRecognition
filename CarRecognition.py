import cv2
import streamlit as st
import numpy as np
from PIL import Image


def main_loop():
    st.title("Car Recognition App")


    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    car_image = Image.open(image_file)
    car_image = np.array(car_image)


if __name__ == '__main__':
    main_loop()