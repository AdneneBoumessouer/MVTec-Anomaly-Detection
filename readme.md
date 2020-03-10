## NOTE: The work on this project is still in progress.

# Anomaly Datection

This project aims at developping a Deep Learning model using an unsupervided method to detect surface anomalies on images.

## Dataset

The dataset being used is the [MVTec dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

## Method

The method being used in this project is inspired to a great extent by the papers [MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf) and [Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders](https://arxiv.org/abs/1807.02011).
The method is devided in 3 steps: training, validating and testing.

## Prerequisites

### Dependencies
Create a new conda environment, activate it and run the following command in your terminal or anaconda prompt to install required libraries and packages used in this project: 
```
conda install --file requirements.txt
```

### Download the Dataset
1. Download the mvtec dataset [here](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and save it to a directory of your choice (e.g in /Downloads)
2. Extract the compressed files.
3. Create a folder named **mvtec** in the project directory.
4. Move the extracted files (contaied in folders) to the **mvtec** folder.


Directory Structure using mvtec dataset
------------
For the scripts to work propoerly, it is required to have a specific directory structure. 
In the case of using the *mvtec* dataset, here is an example of how the directory stucture should look like:

    ├── bottle
    │   ├── ground_truth
    │   │   ├── broken_large
    │   │   ├── broken_small
    │   │   └── contamination
    │   ├── test
    │   │   ├── broken_large
    │   │   ├── broken_small
    │   │   ├── contamination
    │   │   └── good
    │   └── train
    │       └── good
    ├── cable
    │   ├── ground_truth
    │   │   ├── bent_wire
    │   │   ├── cable_swap
    │   │   ├── combined
    │   │   ├── cut_inner_insulation
    │   │   ├── cut_outer_insulation
    │   │   ├── missing_cable
    │   │   ├── missing_wire
    │   │   └── poke_insulation
    │   ├── test
    │   │   ├── bent_wire
    │   │   ├── cable_swap
    │   │   ├── combined
    │   │   ├── cut_inner_insulation
    │   │   ├── cut_outer_insulation
    │   │   ├── good
    │   │   ├── missing_cable
    │   │   ├── missing_wire
    │   │   └── poke_insulation
    │   └── train
    │       └── good
    ...


--------

Directory Structure using your own dataset
------------
To train with your own dataset, you need to have a comparable directory structure. For example:

    ├── class1
    │   ├── test
    │   │   ├── good
    │   │   ├── defect
    │   └── train
    │       └── good
    ├── class2
    │   ├── test
    │   │   ├── good
    │   │   ├── defect
    │   └── train
    │       └── good
    ...


--------

## Training (train.py)

The method uses a Convolutional Auto-Encoder (CAE). There are two proposed variants:
* CAE proposed in the paper [MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf)
* CAE that uses Keras's inception_resnet_v2 CNN-model pretrained on imagenet as the Encoder. 
The Decoder is inspired by the paper [Anomaly Detection and Localization in Images using Guided Attention](https://openreview.net/forum?id=B1gikpEtwH)

During training, the CAE trains exclusively on defect-free images and learns to reconstruct defect-free training samples.

To initiate training, go to the project directory and run the following command in your terminal:
```
python3 train.py new -d <direcroty containing training images> -i <number of training instances> -a <architecture of the model to use> -b <batch size> -l <loss function> -c <color mode> -t <tag>
```
Here is an example:
```
python3 train.py new -d mvtec/capsule -a mvtec -b 12 -l ssim -c grayscale -t "first try"
```
**NOTE:** The number of training epochs will be determined by the given argument `<number of training instances>` specified during the initiation of the training, devided by the actual number of images contained in the train folder of the chosen class.
Example: if you passed `-i 10000` and the training directory contains `1000` images, then the number of epochs will be equal to `10`.

## Finetuning (finetune.py)
This script aims at guiding the the user in chosing the right minimum defect area during validation and test.
It creates the following plots:
* Number of defective regions with increasing thresholds
* Mean and standard deviation of area's size with increasing thresholds
* Distribution of the size of anomaly areas with validation ResMaps for Threshold = 0
* Distribution of the size of anomaly areas with validation ResMaps for increasing Thresholds
* Sample validation image alongside its reconstruction and the resulting ResMap
* Sample test image alongside its reconstruction and the resulting ResMap

To initiate finetuning, go to the project directory and run the following command in your terminal:
```
python3 validate.py -p <path to trained model> -v <sample validation image> -t <sample test image> -r <range of area size to plot>
```

Here is an example:
```
python3 finetune.py -p saved_models/mvtec/capsule/mvtec2/MSE/25-02-2020_08:54:06/CAE_mvtec2_b12.h5 -v "good/000.png" -t "poke/000.png" -r 50
```


## Validation (validate.py)

This script computes the best threshold to use in classification of defect or defect-free images during testing using a small portion (10%) of defect-free training images. To determine a threshold, a minimum defect area that an anomalous region must have to be classified as a defective region must be passed as an argument.

To initiate validation, go to the project directory and run the following command in your terminal:
```
python3 validate.py -p <path to trained model> -a <minimum defective area>
```
Here is an example:
```
python3 validate.py -p saved_models/mvtec/capsule/mvtec2/MSE/25-02-2020_08:54:06/CAE_mvtec2_b12.h5 -a 50
```

## Testing (test.py)

During testing, the classification on the test images takes place either using a threshold and a minimum defect area. You can use the optimal threshold computed during validation or use a combination of threshold and minimum area on your own.

To initiate testing, go to the project directory and run the following command in your terminal:
```
python3 test.py -p <path to trained model> -t <threshold> -a <area>
```
Here is an example using passed arguments for threshold and area:
```
python3 test.py -p saved_models/mvtec/capsule/mvtec2/MSE/25-02-2020_08:54:06/CAE_mvtec2_b12.h5 -t 28 -a 50
```

Project Organization
------------

    ├── mvtec                       <- folder containing all mvtec classes
    │   ├── bottle                  <- subfolder of a class (contains additional subfolders /train and /test)
    |   |── ...
    ├── custom_loss_functions.py    <- implments different loss functions to use in training    
    ├── readme.md                   <- readme file
    ├── requirements.txt            <- requirement text file containing all used packages and libraries
    ├── saved_models                <- directory containing saved models, validation and results
    |   ...
    │       ├── L2                  <- saved models that trained with L2 loss
    │       ├── MSE                 <- saved models that trained with MSE loss
    │       ├── MSSIM               <- saved models that trained with MSSIM loss
    │       └── SSIM                <- saved models that trained with SSIM loss
    ├── models.py                   <- contains different CAE architectures for training
    ├── train.py                    <- training script
    ├── test.py                     <- test script    
    ├── finetune.py                 <- creates plots to help pick a good minimum area for validation
    ├── validate.py                 <- determines best threshold to use when given an area size
    ├── utils.py                    <- utilitary and helper functions    
    └── visualize.py                <- plot and visualize images at different stages (still in progress)


--------

## Authors

* **Adnene Boumessouer** - (https://github.com/AdneneBoumessouer)

See also the list of [contributors](https://github.com/AdneneBoumessouer/Anomaly-Detection/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Paul Bergmann
* Aurélien Geron
* François Chollet


