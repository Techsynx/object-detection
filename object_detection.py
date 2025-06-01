import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time

# Load the trained model
model = YOLO('best.pt')  # Path to your trained YOLOv8 model

st.title("YOLOv8 Object Detection App")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Run detection
    results = model(image_cv, conf=0.3)

    # Annotate image
    annotated_image = results[0].plot()

    st.image(annotated_image, caption="Detected Objects", channels="BGR", use_column_width=True)

st.markdown("---")

# Webcam detection
st.subheader("Live Webcam Detection")
run = st.checkbox('Start Webcam Detection')
FRAME_WINDOW = st.image([])

# Capture webcam
camera = cv2.VideoCapture(0)

if run:
    while True:
        success, frame = camera.read()
        if not success:
            st.error("Failed to read from webcam.")
            break

        # Resize for performance (optional)
        frame_resized = cv2.resize(frame, (640, 480))

        # Run detection
        results = model(frame_resized, conf=0.3)

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        # Show in Streamlit
        FRAME_WINDOW.image(annotated_frame, channels="BGR")

        # Control frame rate (reduce CPU load)
        time.sleep(0.03)
else:
    st.write('Webcam stopped.')
    camera.release()
