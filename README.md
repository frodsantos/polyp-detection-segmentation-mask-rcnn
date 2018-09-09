## Automatic polyp detection, segmentation and classification with Mask R-CNN

The codes are based on implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) by [Matterport, Inc](https://github.com/matterport/Mask_RCNN) on Python 3, Keras, and TensorFlow.


## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 

## Training on Polyp data

Pre-trained weights for MS COCO are provided to make it easier to start. You can use those weights as a starting point to train your own variation on the network. When selecting ```--weights=coco```, if the weights are not downloaded yet it will automatically download them.
Training and test data can be downloaded from: https://giana.grand-challenge.org/Home/
Training and test data directory is set up as follows:

   ```bash
   .
   ├── ...
   ├── new_data                                       # Challenge data
   │   ├── segmentation                               # segmentation datasets
   │   │    ├── train                                 # Training data
   │   │    │     ├── bbdd                            # CVC-300 dataset
   │   │    │     ├── SegmentationTrainingUpload      # CVC-ClinicHDSegment-train dataset with gt
   │   │    │     ├── gtpolyp                         # CVC-300 gt frames
   │   │    ├── test                                  # Testing data
   │   │          ├── CVC612                          # CVC-612 dataset
   │   │          ├── CVC612gt                        # CVC-612 gt frames
   │   │          ├── hdtest                          # CVC-ClinicHDSegment-test dataset with no gt                      
   └── ...
   ```


## 1. Polyp segmentation subsection

Training and evaluation code is in ```code/giana_segmentation.py```. The training schedule, learning rate, and other parameters should be set in this file.

To explore the dataset, you can use this jupyter notebook: ```code/inspect_giana_segmentation_data.ipynb```

To explore the trained model, you can use this jupyter notebook: ```code/inspect_giana_segmentation_model.ipynb```

To train the model you can run it directly from the command line as such:

   ```bash
   # Train a new model starting from pre-trained COCO weights
   python giana_segmentation.py train --dataset=./../new_data/segmentation/train --subset=train --weights=coco
   
   # Train a new model starting from ImageNet weights
   python giana_segmentation.py train --dataset=./../new_data/segmentation/train --subset=train --weights=imagenet

   # Continue training a model that you had trained earlier
   python giana_segmentation.py train --dataset=./../new_data/segmentation/train --subset=train --weights=/path/to/weights.h5

   # Continue training the last model you trained. This will find
   # the last trained weights in the model directory.
   python giana_segmentation.py train --dataset=./../new_data/segmentation/train --subset=train --weights=last
   ```


You can predict on the test by running the code below. Output results will be stored in ```results/segmentation_sd/detection_submit_xxx``` and ```results/segmentation_hd/localisation_submit_xxx```
   
   ```bash
   # Run evaluation on the last trained model
   python giana_segmentation.py detect --dataset=./../new_data/segmentation/test --subset=test --weights=/path/to/weights.h5
   ```


## 2. Requirements

Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in ```requirements.txt```.
