import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import scipy.io

from config import Config
import utils
import model as modellib
import visualize
from model import log

#%matplotlib inline

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs.nosync")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5.nosync")

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "layout"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 11  # background + layouts

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

config = ShapesConfig()
config.display()


class LayoutDataset(utils.Dataset):

    #load .mat layout information files:
    training_layout_info = {}
    validation_layout_info = {}
    training_mat = scipy.io.loadmat('layout/training.mat')
    for i in training_mat['training'].flatten():
        training_layout_info.update({i['image'][0]:{'scene':i['scene'][0],'type':i['type'][0][0],'point':i['point'],'resolution':i['resolution'][0]}})
    validation_mat = scipy.io.loadmat('layout/validation.mat')
    for i in validation_mat['validation'].flatten():
        validation_layout_info.update({i['image'][0]:{'scene':i['scene'][0],'type':i['type'][0][0],'point':i['point'],'resolution':i['resolution'][0]}})
    training_img_name_list = []
    validation_img_name_list = []
    test_img_name_list = []
    for i in os.listdir('layout/images'):
        if i[:-4] in training_layout_info.keys():
            training_img_name_list.append(i[:-4])
        elif i[:-4] in validation_layout_info.keys():
            validation_img_name_list.append(i[:-4])
        else: test_img_name_list.append(i[:-4])

    #set mask point size
    mask_point_size = 20

    def load_layouts(self,type):
        #11 kinds of layouts
        self.add_class("layouts", 1, "type0")
        self.add_class("layouts", 2, "type1")
        self.add_class("layouts", 3, "type2")
        self.add_class("layouts", 4, "type3")
        self.add_class("layouts", 5, "type4")
        self.add_class("layouts", 6, "type5")
        self.add_class("layouts", 7, "type6")
        self.add_class("layouts", 8, "type7")
        self.add_class("layouts", 9, "type8")
        self.add_class("layouts", 10, "type9")
        self.add_class("layouts", 11, "type10")
        if type == 'training':
            for img_id in range(len(self.training_img_name_list)):
                img_name = self.training_img_name_list[img_id]
                img_info = self.get_info(img_name)
                self.add_image("layouts", image_id=img_id, path=
                               'layout/images/'+img_name+'.jpg',
                               name=img_name, scene=img_info['scene'],
                               point=img_info['point'],
                               resolution=img_info['resolution'],
                               type=img_info['type'])
        else:
            for img_id in range(len(self.validation_img_name_list)):
                img_name = self.validation_img_name_list[img_id]
                img_info = self.get_info(img_name)
                self.add_image("layouts", image_id=img_id, path=
                               'layout/images/'+img_name+'.jpg',
                               name=img_name, scene=img_info['scene'],
                               point=img_info['point'],
                               resolution=img_info['resolution'],
                               type=img_info['type'])

    def get_info(self, image_name):
        """Input image_id, output layout info"""
        if image_name in self.validation_layout_info.keys():
            return self.validation_layout_info[image_name]
        elif image_name in self.training_layout_info.keys():
            return self.training_layout_info[image_name]
        else:
            print("image name doesn't exist")
            return False

    def image_reference(self, image_id):
        """Return the layouts data of the image."""
        return  self.image_info[image_id]

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        class_ids = np.array([info['type'] + 1], dtype=np.int32)
        mask = np.resize(np.array([([0]*info['resolution'][1]) for i in range(info['resolution'][0])], dtype=np.uint8),(info['resolution'][0],info['resolution'][1],1))
        for point in info['point']:
            mask[max(0,int(point[1]-0.5)-int(self.mask_point_size/2)):
                 min(info['resolution'][0],int(point[1]-0.5)+int(self.mask_point_size/2)),
            max(0,int(point[0]-0.5)-int(self.mask_point_size/2)):
                 min(info['resolution'][1],int(point[0]-0.5)+int(self.mask_point_size/2)),0] = 1
        return mask, class_ids


dataset_train = LayoutDataset()
dataset_train.load_layouts('training')
dataset_train.prepare()
dataset_val = LayoutDataset()
dataset_val.load_layouts('validation')
dataset_val.prepare()
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')
