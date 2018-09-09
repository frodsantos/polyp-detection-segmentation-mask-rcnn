# -*- coding: utf-8 -*-

"""
Mask R-CNN - Configuration for GIANA challenge, segmentation SD and HD

    Code adapted from:
        https://github.com/matterport/Mask_RCNN/blob/master/samples/nucleus/nucleus.py
        
        
        Copyright (c) 2018 Matterport, Inc.
        Licensed under the MIT License (see LICENSE for details)
        Written by Waleed Abdulla
        ------------------------------------------------------------
"""

# Set matplotlib backend
# This has to be done before other imports that might
# set it, but only if we are running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

import os
import re
import sys
import csv
import cv2
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory. Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Validation images
VAL_IMAGE_IDS = ["bbdd_1.bmp", "bbdd_9.bmp", "bbdd_22.bmp", "bbdd_30.bmp", "bbdd_39.bmp", "bbdd_60.bmp",
                 "bbdd_64.bmp", "bbdd_74.bmp", "bbdd_93.bmp", "bbdd_89.bmp", "bbdd_107.bmp", "bbdd_103.bmp",
                 "bbdd_120.bmp", "bbdd_141.bmp", "bbdd_151.bmp", "bbdd_162.bmp", "bbdd_169.bmp", "bbdd_179.bmp",
                 "bbdd_199.bmp", "bbdd_206.bmp", "bbdd_211.bmp", "bbdd_222.bmp", "bbdd_227.bmp", "bbdd_235.bmp",
                 "bbdd_248.bmp", "bbdd_251.bmp", "bbdd_266.bmp", "bbdd_276.bmp", "bbdd_293.bmp", "bbdd_296.bmp",
                 "SegmentationTrainingUpload_27.bmp", "SegmentationTrainingUpload_40.bmp",
                 "SegmentationTrainingUpload_13.bmp", "SegmentationTrainingUpload_37.bmp",
                 "SegmentationTrainingUpload_54.bmp", "SegmentationTrainingUpload_46.bmp"]

############################################################
#  Configurations
############################################################

class PolypsConfig(Config):
    """Configuration for training on the polyp segmentation dataset."""

    # Name the configurations
    NAME = "segmentation"

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 6

    # Number of training steps per epoch
    # This does not need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so do not set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 354 // IMAGES_PER_GPU

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = max(1, 36 // IMAGES_PER_GPU)

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classification classes (including background)
    NUM_CLASSES = 1 + 1 # polyp mask + background

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024)
    
    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.8

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 2000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and inferencing
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling does not make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "crop" # Random crops of size 1024x1024
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM does not require it.
    # Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0.

    # Image mean (RGB), obtained from dataset statistics.
    MEAN_PIXEL = np.array([111.35, 72.49, 50.83])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN does not generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 10

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 10

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    # Do not exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.0

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.2

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.5
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (do not use). Set layer in training mode even when inferencing
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0
        
    # Mask binarization threshold
    DETECTION_MASK_THRESHOLD = 0.30
    
    
class PolypsInferenceConfig(PolypsConfig):
    """Configuration for inference on the polyp segmentation dataset."""
    
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Do not resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    # Max number of final detections
    DETECTION_MAX_INSTANCES = 1
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.0


############################################################
#  Dataset
############################################################

class PolypsDataset(utils.Dataset):

    def load_polyps(self, dataset_dir, subset):
        """Load a subset of the polyps dataset.
        """        
        # Add classes. We have one class.
        # Naming the dataset polyps, and the class polyps
        self.add_class("polyps", 1, "polyps")

        # Which subset?
        assert subset in ["train", "val", "test", "SegmentationTrainingUpload", "bbdd", "hdtest", "CVC612"]
        
        # If the subset is either train or validation
        # we will use all the subfolders specified in data_folders
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        elif subset == "train":
            subfolders = ["bbdd", "SegmentationTrainingUpload"]
        elif subset == "test":
            subfolders = ["CVC612", "hdtest"]
        else:
            subfolders = [subset]
        
        if subset != "val":
            image_ids = []
            for subfolder in subfolders:
                # Add all images in the subfolder and assign image_id
                images_dir = os.path.join(dataset_dir, subfolder)
                images_subfolder = [(subfolder + "_" + image) for image in next(os.walk(images_dir))[2] if "mask" not in image]
                
                # Extend the lists with image IDs
                image_ids.extend(images_subfolder)

            # Final training done on all data, hence these two lines are commented.
             if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add images
        for image_id in image_ids:
            image_path = image_id.split("_")

            self.add_image(
                    "polyps",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, image_path[0], image_path[1]))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.dirname(info['path'])

        # Read mask files from .png image
        mask = []
        # There is only one mask per image in this dataset
        full_image_name, image_format = info['id'].split('.')
        image_folder, image_name = full_image_name.split("_")
        if image_folder == "bbdd":
            mask_dir = os.path.join(os.path.dirname(mask_dir), "gtpolyp")
            mask_name = image_name + "." + image_format
        elif image_folder == "CVC612":
            mask_dir = os.path.join(os.path.dirname(mask_dir), "CVC612gt")
            mask_name = image_name + "." + image_format
        else:
            mask_name = image_name + "_mask.tif"

        m = skimage.io.imread(os.path.join(mask_dir, mask_name)).astype(np.bool)
        mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "polyps":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = PolypsDataset()
    dataset_train.load_polyps(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PolypsDataset()
    dataset_val.load_polyps(dataset_dir, "val")
    dataset_val.prepare()
    
    
    # Current data augmentation
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sometimes(0.5,
                      [iaa.Affine(rotate=(-45, 45)),
                       iaa.Affine(rotate=(-90, 90)),
                       iaa.Affine(scale=(0.5, 1.5)),
                       iaa.Affine(shear=(-16, 16))]),
        iaa.SomeOf((0, 1),
                       [iaa.CropAndPad(percent=(-0.25, 0.25)),
                       iaa.Multiply((0.8, 1.5)),
                       iaa.GaussianBlur(sigma=(0.0, 5.0)),
                       iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
                       iaa.WithChannels(0, iaa.Affine(rotate=(0, 10)))])
    ])

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                augmentation=augmentation,
                layers='all')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/10,
                epochs=60,
                augmentation=augmentation,
                layers='all')
                
############################################################
#  Segmentation
############################################################

def create_submission_dirs(subfolder):
    # Create directory
    submit_folder = os.path.join(RESULTS_DIR, subfolder)
    if not os.path.exists(submit_folder):
        os.makedirs(submit_folder)
    submit_dir = subfolder + "_submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(submit_folder, submit_dir)
    os.makedirs(submit_dir)
    
    return submit_dir
    
def save_submission(pred_masks, submit_dir, image_num):

    predicted_mask = np.ufunc.reduce(np.logical_or, pred_masks, axis=2)
    file_path = os.path.join(submit_dir, str(image_num) + "_mask.png")
    cv2.imwrite(file_path, predicted_mask.astype('uint8') * 255)
    
    print("Saved to ", file_path)
    
def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    sd_submit_dir = create_submission_dirs("segmentation_sd")
    hd_submit_dir = create_submission_dirs("segmentation_hd")

    # Read dataset
    dataset = PolypsDataset()
    dataset.load_polyps(dataset_dir, subset)
    dataset.prepare()
    
    # Run detection for each image in the dataset
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        
        source_id = re.findall("([0-9]+)[^-]", dataset.image_info[image_id]["id"])[0]
        image_folder, image_name = dataset.image_reference(image_id).split("_")
        if image_folder == "hdtest":
            save_submission(r['masks'], hd_submit_dir, image_name.split(".")[0])
        else:
            save_submission(r['masks'], sd_submit_dir, image_name.split(".")[0])

############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for polyps detection and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PolypsConfig()
    else:
        config = PolypsInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
