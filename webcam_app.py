import os
import cv2
import numpy as np
import streamlit as st
import insightface
from insightface.app import FaceAnalysis
from insightface.data  import get_image as ins_get_image
from glob import glob
from tqdm import tqdm
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import shutil
import zipfile
import image_app
import utils
from utils import app



class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.names = None
        self.embeddings = None

    def _recognize_faces(self, frame):
        if self.names is None or self.embeddings is None:
            return frame

        # Perform face analysis on the frame
        faces = self.app.get(frame)

        # Process each detected face separately
        for face in faces:
            # Retrieve the embedding for the detected face
            detected_embedding = face.normed_embedding

            # Calculate similarity scores with known embeddings
            scores = np.dot(detected_embedding, np.array(self.embeddings).T)
            scores = np.clip(scores, 0., 1.)

            # Find the index with the highest score
            idx = np.argmax(scores)
            max_score = scores[idx]

            # Check if the maximum score is above a certain threshold (adjust as needed)
            threshold = 0.7
            if max_score >= threshold:
                recognized_name = self.names[idx]
            else:
                recognized_name = "Unknown"

            # Draw bounding box around the detected face
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            # Write recognized name within the bounding box
            cv2.putText(frame, recognized_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Debug print
            print("Detected face:", recognized_name, "with confidence:", max_score)

        return frame

    def transform(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self._recognize_faces(frame)
        return frame