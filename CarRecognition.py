import cv2
import streamlit as st
import numpy as np
import io
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img
import scipy.io

def load_and_preprocess_image(image_bytes):
    img = load_img(io.BytesIO(image_bytes), target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    return img

def load_car_labels_and_bboxes():
    # Load the annotations from cars_annos.mat
    annotations = scipy.io.loadmat('cars_annos.mat')
    annotations = annotations['annotations'][0]

    car_labels = [str(annotation[5][0]) for annotation in annotations]
    bboxes = [(int(annotation[0][0]), int(annotation[1][0]), int(annotation[2][0]), int(annotation[3][0])) for annotation in annotations]
    car_names = [str(annotation[4][0]) for annotation in annotations]

    return car_labels, bboxes, car_names

def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def main_loop():
    st.title("Car Recognition App")

    model = create_model()
    train_path = 'cars_train/'
    test_path = 'cars_test'

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    image_bytes = image_file.read()
    car_image = load_and_preprocess_image(image_bytes)

    # Prediction
    prediction = model.predict(np.expand_dims(car_image, axis=0))

    # Decode predictions and display the top result
    car_model_labels = ["Model_1", "Model_2", "Model_3", ...]

    predicted_class = np.argmax(prediction)
    car_model = car_model_labels[predicted_class]
    
    # Debugging statements
    print(f"Predicted Class: {predicted_class}")
    print(f"Length of car_model_labels: {len(car_model_labels)}")

    # Check if predicted_class is a valid index
    if 0 <= predicted_class < len(car_model_labels):
      car_model = car_model_labels[predicted_class]
      st.image(image_bytes, channels="RGB", use_column_width=True)
      st.write(f"Predicted Car Model: {car_model}")
    else:
      st.write("Invalid prediction index.")

    st.image(image_bytes, channels="RGB", use_column_width=True)
    st.write(f"Predicted Car Model: {car_model}")

    # Retrieve the car's bounding box and name from the annotations
    car_labels, bboxes, car_names = load_car_labels_and_bboxes()
    bounding_box = bboxes[predicted_class]
    car_name = car_names[predicted_class]

    st.write(f"Car Name: {car_name}")
    st.write(f"Bounding Box (x, y, width, height): {bounding_box}")

if __name__ == '__main__':
    main_loop()