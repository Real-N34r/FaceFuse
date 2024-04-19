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

def face_recognition_in_video(video_path, output_path, names, embeddings):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame dimensions and frame rate of the input video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform face analysis on the frame
        faces = app.get(frame)

        # Process each detected face separately
        for face in faces:
            # Retrieve the embedding for the detected face
            detected_embedding = face.normed_embedding

            # Calculate similarity scores with known embeddings
            scores = np.dot(detected_embedding, embeddings.T)
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
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            # Write recognized name within the bounding box
            cv2.putText(frame, recognized_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the frame into the output video
        out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    return output_path
