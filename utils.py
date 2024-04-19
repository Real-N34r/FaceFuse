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


app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640),)

