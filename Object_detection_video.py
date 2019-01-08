######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import ctypes
from random import randint

user32 = ctypes.windll.user32
SCREEN_WIDTH = user32.GetSystemMetrics(0)
SCREEN_HEIGHT = user32.GetSystemMetrics(1)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'test_video.avi'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)
# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
     
if int(major_ver)  < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

frame_counter = 0

time = []
colors = []
alerts = []
# Alert levels

time.append(10)
time.append(20)
time.append(30)
time.append(40)

colors.append((17, 212, 76))   # b g r   alert 1 - Acceptable 
colors.append((46, 181, 238))  # b g r   alert 2 - Email to Level 1
colors.append((0, 132, 255))   # b g r   alert 3 - Email to Level 2
colors.append((0, 0, 255))     # b g r   alert 4 - Email to Level 3

for i in range(len(colors)):
    alerts.append((colors[i], time[i]))

isFirstCapture = True
while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    if (isFirstCapture):
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        
        height, width, channels = frame.shape       # screen size
        isFirstCapture = False

    min_thresh_score = 0.80
    # Draw the results of the detection (aka 'visualize the results')
    #vis_util.visualize_boxes_and_labels_on_image_array(
    #    frame,
    #    np.squeeze(boxes),
    #    np.squeeze(classes).astype(np.int32),
    #    np.squeeze(scores),
    #    category_index,
    #    use_normalized_coordinates=True,
    #    line_thickness=4,
    #    min_score_thresh=min_thresh_score)

    scoress = np.squeeze(scores)
    boxx = np.squeeze(boxes)
    counter = 0
    for x in range(len(scoress)):
        if(scoress[x] > min_thresh_score):
            ymin, xmin, ymax, xmax = boxx[x]
            ymin = int(ymin*height)
            ymax = int(ymax*height)
            xmin = int(xmin*width)
            xmax = int(xmax*width)
            
            p1 = xmin, ymin
            p2 = xmax, ymax
            
            if(frame_counter < alerts[0][1]*25):
                i = 0
            elif(frame_counter >= alerts[0][1]*25 and frame_counter < alerts[1][1]*25):
                i = 1
            elif(frame_counter >= alerts[1][1]*25 and frame_counter < alerts[2][1]*25):
                i = 2
            elif(frame_counter > alerts[2][1]*25): 
                i = 3

            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            counter = counter + 1
        
    # All the results have been drawn on the frame, so it's time to display it.
    # demo = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
    
    cv2.imshow('Object detector', frame)

    frame_counter = frame_counter + 1
    print(frame_counter)
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
