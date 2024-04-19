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
import tempfile
import os
import image_app
from utils import app
from embeddings_app import get_embeddings
import webcam_app
from webcam_app import FaceRecognitionTransformer
import video_app
from video_app import face_recognition_in_video

# Function to extract zip file
def extract_zip(zip_file_path, extract_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


# Function to delete files and directory
def delete_files(db_dir):
    shutil.rmtree(db_dir)

# Main function
def main():
    st.title("FaceFuse")
    # Tabs
    tabs = ["Embeddings", "Face Recognition in Image", "Face Recognition in Video", "Webcam"]
    choice = st.sidebar.selectbox("Select Option", tabs)

    # Embeddings tab
    if choice == "Embeddings":
        st.subheader("Upload a Zip File")
        uploaded_file = st.file_uploader("Choose a zip file", type="zip")

        if uploaded_file is not None:
            with open("temp.zip", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File uploaded successfully!")
            extract_zip("temp.zip", "temp")  # Extract the uploaded zip file to the temp directory

        if st.button("Get Embeddings"):
            get_embeddings("temp")
            st.success("Embeddings generated successfully!")

        if st.button("Download names.npy"):
            with open("temp/names.npy", "rb") as file:
                st.download_button("Download names.npy", file.getvalue(), file_name="names.npy", mime="application/octet-stream")
    
        if st.button("Download embeddings.npy"):
            with open("temp/embeddings.npy", "rb") as file:
                st.download_button("Download embeddings.npy", file.getvalue(), file_name="embeddings.npy", mime="application/octet-stream")

        if st.button("Delete Files"):
            delete_files("temp")
            st.success("Files deleted successfully!")

    # Other tabs can be added similarly
    if choice == "Webcam":
        st.header("WEBCAM")
        st.subheader("Upload names and embeddings file")
        uploaded_names = st.file_uploader("Upload names.npy", type="npy")
        uploaded_embeddings = st.file_uploader("Upload embeddings.npy", type="npy")

        if uploaded_names and uploaded_embeddings:
            names = np.load(uploaded_names)
            embeddings = np.load(uploaded_embeddings)

            # Initialize transformer with names and embeddings
            transformer = FaceRecognitionTransformer()
            transformer.names = names
            transformer.embeddings = embeddings

            # Create WebRTC streamer
            webrtc_ctx = webrtc_streamer(
                key="example",
                video_processor_factory=FaceRecognitionTransformer,
                async_processing=True,
            )

    if choice == "Face Recognition in Image":
        st.header("Image Recognition")
        st.subheader("Upload names and embeddings file")
        upload_names = st.file_uploader("Upload names.npy", type="npy")
        upload_embeddings = st.file_uploader("Upload embeddings.npy", type="npy")
        st.subheader("Upload Image")
        upload_img = st.file_uploader("Upload Image",type=["png","jpg"])

        if upload_img and upload_names and upload_embeddings:
            names = np.load(upload_names)
            embeddings = np.load(upload_embeddings)
            im_array = np.frombuffer(upload_img.read(),np.uint8)
            img = cv2.imdecode(im_array,cv2.IMREAD_COLOR)
            if st.button("Verify Faces"):
                image_app.recognize_and_display(img,embeddings,names,app)

    #video_recognition
    if choice == "Face Recognition in Video": 
        st.header("Face Recognition in Video")
        st.subheader("Upload Video and NPY Files")
        temp_dir = tempfile.mkdtemp()
        upload_names_1 = st.file_uploader("Upload names.npy", type="npy")
        upload_embeddings_1 = st.file_uploader("Upload embeddings.npy", type="npy")
        st.subheader("Upload Video")
        upload_video = st.file_uploader("Upload Video",type=["mp4"])
        if upload_video and upload_names_1 and upload_embeddings_1:
            video_path = os.path.join(temp_dir, "input_video.mp4")
            names_path = os.path.join(temp_dir, "names.npy")
            embeddings_path = os.path.join(temp_dir, "embeddings.npy")

            # Save uploaded files to temporary directory
            with open(video_path, "wb") as video_file:
                video_file.write(upload_video.read())

            with open(names_path, "wb") as names_file:
                names_file.write(upload_names_1.read())

            with open(embeddings_path, "wb") as embeddings_file:
                embeddings_file.write(upload_embeddings_1.read())

            names = np.load(names_path)
            embeddings = np.load(embeddings_path)

            # Verify Faces button
            if st.button("Verify Faces"):
                output_path = os.path.join(temp_dir, "output_video.mp4")
                # Run face recognition in video function
                output_video_path = face_recognition_in_video(video_path, output_path, names, embeddings)
                st.write("Face recognition completed!")

                # Download button for the output video
                with open(output_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.download_button(label="Download Processed Video", data=video_bytes, file_name="output_video.mp4", mime="video/mp4")



if __name__ == "__main__":
    main()
