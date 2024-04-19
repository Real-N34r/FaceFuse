import streamlit as st
import cv2
import numpy as np
import os
from tqdm import tqdm
from glob import glob
import numpy as np
import cv2 as cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data  import get_image as ins_get_image
import cv2
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import utils
from utils import app


# Define function to recognize faces and display bounding boxes
def recognize_and_display(input_img, known_embeddings, names, app):
    # Perform face analysis on the input image
    faces = app.get(input_img)

    # Check if any face is detected
    if len(faces) == 0:
        # If no face detected, draw bounding box with "unknown" text for the whole image
        input_img_with_bb = input_img.copy()
        cv2.putText(input_img_with_bb, "Unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        st.image(cv2.cvtColor(input_img_with_bb, cv2.COLOR_BGR2RGB), caption='No face detected', use_column_width=True)
        return "No face detected"
    else:
        # Process each detected face separately
        for face in faces:
            # Retrieve the embedding for the detected face
            detected_embedding = face.normed_embedding

            # Calculate similarity scores with known embeddings
            scores = np.dot(detected_embedding, np.array(known_embeddings).T)
            scores = np.clip(scores, 0., 1.)

            # Find the index with the highest score
            idx = np.argmax(scores)
            max_score = scores[idx]

            # Check if the maximum score is above a certain threshold (adjust as needed)
            threshold = 0.7
            if max_score >= threshold:
                recognized_name = names[idx]
            else:
                recognized_name = "Unknown"

            # Draw bounding box around the detected face
            bbox = face.bbox.astype(int)
            cv2.rectangle(input_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 10)
            # Write recognized name within the bounding box
            cv2.putText(input_img, recognized_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 0), 10)

        # Display the image with bounding boxes using Streamlit
        st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), caption='Face Recognition Result', use_column_width=True)
        output_img_bytes = cv2.imencode(".jpg", input_img)[1].tobytes()
        st.download_button("Download Output Image", output_img_bytes, file_name="output_image.jpg", mime="image/jpeg")

        if "Unknown" in names:
            return "Face not recognized"
        else:
            return f"All faces recognized"
