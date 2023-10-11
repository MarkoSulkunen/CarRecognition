import cv2
import streamlit as st
import numpy as np
import io

def main_loop():
    st.title("Car Recognition App")
    
    car_cascade = cv2.CascadeClassifier('cars.xml')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    # Read the image file as bytes
    image_bytes = image_file.read()
    # Use io.BytesIO to convert the bytes-like object to a readable stream
    image_stream = io.BytesIO(image_bytes)
    # Read the image using OpenCV
    car_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    
    img_gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
    
    found = car_cascade.detectMultiScale(img_gray, minSize=(20, 20))

    # Draw rectangles around recognized cars and add text labels
    for i, (x, y, width, height) in enumerate(found, 1):
        cv2.rectangle(car_image, (x, y), (x + width, y + height), (0, 255, 0), 5)
        text = f"Car {i}"
        cv2.putText(car_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the image using Streamlit
    st.image(car_image, channels="BGR", use_column_width=True)

if __name__ == '__main__':
    main_loop()