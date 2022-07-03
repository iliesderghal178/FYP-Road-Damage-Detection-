#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import sys
import tarfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
import cv2

import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# if tf.__version__ != '1.4.1':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.1!')
from utils import label_map_util

from utils import visualization_utils as vis_util
from detection_utils import parse_output_dict


# In[41]:


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT =  r"C:\Users\telomere\Desktop\FYP\ssd_mobilenet_v2_coco.config"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = r"C:\Users\telomere\Desktop\FYP\RoadDamageDetector\crackLabelMap.txt"

NUM_CLASSES = 8


# In[42]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# In[43]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# In[44]:


def load_image_into_numpy_array(image):
    print(image.size)
    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# In[6]:


import os


# In[49]:


PATH_TO_TEST_IMAGES_DIR = r'C:\Users\telomere\Desktop\FYP\Data\images'
arr = os.listdir(PATH_TO_TEST_IMAGES_DIR)


# In[50]:


def addition(n):
    
    return  os.path.join(PATH_TO_TEST_IMAGES_DIR,n)
  
# We double all numbers using map()
result = map(addition, arr)
images_test=list(result)


# In[51]:


def modify(output_dict):
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections] 
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    return output_dict


# In[53]:


with detection_graph.as_default():
    print("sadas")
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in images_test[1:]:
            print(image_path)
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            output_dict= {}
            output_dict["detection_boxes"] =boxes
            output_dict['detection_classes'] = classes
            output_dict['detection_scores'] = scores
            output_dict['num_detections'] = num
            output_dict=modify(output_dict)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
                output_dict['detection_boxes'],output_dict['detection_classes'],output_dict['detection_scores'],
             category_index,
               min_score_thresh=0.5,
              use_normalized_coordinates=True,
              line_thickness=8)
            final_image_path = image_path[:-4]+"_det.jpg"
            cv2.imwrite(final_image_path,image_np)
            plt.imshow(image_np)

