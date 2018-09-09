## Polyp detection, segmentation and classification
GIANA challenge 2018

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

## Training on Polyp/WCE data

Pre-trained weights for MS COCO are provided to make it easier to start. You can use those weights as a starting point to train your own variation on the network. When selecting ```--weights=coco```, if the weights are not downloaded yet it will automatically download them.
Training and test data can be downloaded from: https://giana.grand-challenge.org/Home/
Training and test data directory is set up as follows:

   ```bash
   .
   ├── ...
   ├── new_data                                       # Challenge data
   │   ├── wce                                        # wce datasets
   │   │    ├── train                                 # Training data
   │   │    │     ├── inflammatory                    # Inflammatory classification data with gt
   │   │    │     ├── vascularlesions                 # Vascular lesions classification data with gt
   │   │    │     ├── normal                          # Normal (background) classification data
   │   │    ├── test                                  # Testing data, we have no gt for testing data
   │   │          ├── alltest             
   │   ├── segmentation                               # segmentation datasets
   │   │    ├── train                                 # Training data
   │   │    │     ├── bbdd                            # CVC-300 dataset
   │   │    │     ├── SegmentationTrainingUpload      # CVC-ClinicHDSegment-train dataset with gt
   │   │    │     ├── gtpolyp                         # CVC-300 gt frames
   │   │    ├── test                                  # Testing data
   │   │          ├── CVC612                          # CVC-612 dataset
   │   │          ├── CVC612gt                        # CVC-612 gt frames
   │   │          ├── hdtest                          # CVC-ClinicHDSegment-test dataset with no gt
   │   └── detection                                  # detection datasets
   │   │    ├── train                                 # Training data
   │   │    │     ├── 1                               # 18 folders with the training frames including gt
   │   │    │     ...
   │   │    │     ├── 18                            
   │   │    ├── test                                  # Testing data
   │   │    │     ├── 1                               # 18 folders with the test frames. No gt included
   │   │    │     ...
   │   │    │     ├── 18                            
   └── ...
   ```

## 1. Polyp detection and localisation subsections

Training and evaluation code is in ```code/giana_detection.py```. The training schedule, learning rate, and other parameters should be set in this file.

To explore the dataset, you can use this jupyter notebook: ```code/inspect_giana_detection_data.ipynb```

To explore the trained model, you can use this jupyter notebook: ```code/inspect_giana_detection_model.ipynb```

To train the model you can run it directly from the command line as such:

   ```bash
   # Train a new model starting from pre-trained COCO weights
   python giana_detection.py train --dataset=./../new_data/detection/train --subset=train --weights=coco
   
   # Train a new model starting from ImageNet weights
   python giana_detection.py train --dataset=./../new_data/detection/train --subset=train --weights=imagenet

   # Continue training a model that you had trained earlier
   python giana_detection.py train --dataset=./../new_data/detection/train --subset=train --weights=/path/to/weights.h5

   # Continue training the last model you trained. This will find
   # the last trained weights in the model directory.
   python giana_detection.py train --dataset=./../new_data/detection/train --subset=train --weights=last
   ```
   
You can compute evaluation metrics on the validation set with:

   ```bash
   # Run evaluation metrics on the validation for all the saved models
   python metrics.py detection_log_folder_name --dataset=detection
   ```
   
You can predict on the test by running the code below. Output results will be stored in ```results/detection/detection_submit_xxx``` and ```results/localisation/localisation_submit_xxx```
   
   ```bash
   # Run evaluation on the last trained model
   python giana_detection.py detect --dataset=./../new_data/detection/test --subset=test --weights=/path/to/weights.h5
   ```
   
## 2. Polyp segmentation subsection

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
   
You can compute evaluation metrics on the validation set with:

   ```bash
   # Run evaluation metrics on the validation for all the saved models
   python metrics.py detection_log_folder_name --dataset=segmentation
   ```
   
You can predict on the test by running the code below. Output results will be stored in ```results/segmentation_sd/detection_submit_xxx``` and ```results/segmentation_hd/localisation_submit_xxx```
   
   ```bash
   # Run evaluation on the last trained model
   python giana_segmentation.py detect --dataset=./../new_data/segmentation/test --subset=test --weights=/path/to/weights.h5
   ```
   
## 3. WCE detection and localisation subsections

Training and evaluation code is in ```code/giana_wce.py```. The training schedule, learning rate, and other parameters should be set in this file.

To explore the dataset, you can use this jupyter notebook: ```code/inspect_giana_wce_data.ipynb```

To explore the trained model, you can use this jupyter notebook: ```code/inspect_giana_wce_model.ipynb```

To train the model you can run it directly from the command line as such:

   ```bash
   # Train a new model starting from pre-trained COCO weights
   python giana_wce.py train --dataset=./../new_data/wce/train --subset=train --weights=coco
   
   # Train a new model starting from ImageNet weights
   python giana_wce.py train --dataset=./../new_data/wce/train --subset=train --weights=imagenet

   # Continue training a model that you had trained earlier
   python giana_wce.py train --dataset=./../new_data/wce/train --subset=train --weights=/path/to/weights.h5

   # Continue training the last model you trained. This will find
   # the last trained weights in the model directory.
   python giana_wce.py train --dataset=./../new_data/wce/train --subset=train --weights=last
   ```
   
You can compute evaluation metrics on the validation set with:

   ```bash
   # Run evaluation metrics on the validation for all the saved models
   python metrics.py detection_log_folder_name --dataset=wce
   ```
   
You can predict on the test by running the code below. Output results will be stored in ```results/wce_detect``` and ```results/wce_local```
   
   ```bash
   # Run evaluation on the last trained model
   python giana_wce.py detect --dataset=./../new_data/wce/test --subset=test --weights=/path/to/weights.h5
   ```

## 4. Additional commands used

   ```bash
   # Activate environment in dgx
   source ~/tensorflow/tensorflowgpu/bin/activate
   ```
   
   ```bash
   # Set GPU 3 as visible device
   export CUDA_VISIBLE_DEVICES=3
   
   # Checks what GPU is currently visible
   env | grep CUDA
   ```
