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
import streamlit as st
from utils import app

def get_embeddings(db_dir):
    names = []
    embeddings = []

    # Traverse through each subfolder
    for root, dirs, files in os.walk(db_dir):
        for folder in dirs:
            if folder == ".ipynb_checkpoints":
                continue
            img_paths = glob(os.path.join(root, folder, '*'))
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                faces = app.get(img)
                if len(faces) != 1:
                    continue
                face = faces[0]
                names.append(folder)
                embeddings.append(face.normed_embedding)

    if embeddings:
        embeddings = np.stack(embeddings, axis=0)
        np.save(os.path.join(db_dir, "embeddings.npy"), embeddings)
        np.save(os.path.join(db_dir, "names.npy"), names)
    else:
        st.warning("No embeddings generated. Please ensure that there are valid images with detected faces.")