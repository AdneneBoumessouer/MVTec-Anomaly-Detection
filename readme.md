## NOTE: The work on this project is still in progress.

# Anomaly Detection

This project aims at developping a Deep Learning model using an unsupervided method to detect surface anomalies on images.

## Overview
![Image of Yaktocat](overview.png)

The method being used in this project is inspired to a great extent by the papers [MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf) and [Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders](https://arxiv.org/abs/1807.02011).
The method is devided in 3 steps: training, finetuning and testing.

## Dataset

The dataset being used is the [MVTec dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

## Prerequisites

### Dependencies
Libraries and packages used in this project: 
* `tensorflow-gpu 2.1.0`
* `Keras 2.3.1`
* `ktrain 0.13.0`
* `scikit-image 0.17.2`
* `opencv-python 4.2.0.34`
* `pandas 1.0.3`
* `numpy 1.18.1`
* `matplotlib 3.1.3`


### Download the Dataset
1. Download the mvtec dataset [here](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and save it to a directory of your choice (e.g in /Downloads)
2. Extract the compressed files.
3. Create a folder named **mvtec** in the project directory.
4. Move the extracted files to the **mvtec** folder.


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

## Training (`train.py`)

### Description
The method uses a Convolutional Auto-Encoder (CAE). There are two proposed variants:
* CAE proposed in the paper [MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf)
* CAE that uses Keras's inception_resnet_v2 CNN-model pretrained on imagenet as the Encoder. 
The Decoder is inspired by the paper [Anomaly Detection and Localization in Images using Guided Attention](https://openreview.net/forum?id=B1gikpEtwH). NOTE: This model seems not to work properly at the moment and is still being tested.

During training, the CAE trains exclusively on defect-free images and learns to reconstruct (predict) defect-free training samples.

### Usage
usage: train.py [-h] -d  [-a] [-c] [-l] [-b] [-i]

optional arguments:

-h, --help            show this help message and exit

-d , --input-dir      directory containing training images

-a , --architecture   architecture of the model to use for training: 'resnet', 'mvtec' or 'mvtec2'

-c , --color          color mode for preprocessing images before training: 'rgb' or 'grayscale'

-l , --loss           loss function to use for training: 'mssim', 'ssim' or 'l2'

-b , --batch          batch size to use for training

-i, --inspect         generate inspection plots after training


Example usage:
```
python3 train.py -d mvtec/capsule -a mvtec2 -b 8 -l ssim -c grayscale
```
**NOTE 1:** There is no need for the user to pass a number of epochs since the training process implements an Early Stopping strategy.

**NOTE 2:** There is a total of 3 models implemented in this project: *resnet*, *mvtec* and *mvtec2*. *Resnet* seems not to be working properly at the moment and needs further investigation/testing. 

**NOTE 3:** While *mvtec* and *mvtec2* are two slightly different variants of the same model, we **recommend** opting for mvtec2, as it has been tested extensively.


## Finetuning (`finetune.py`)
This script approximates a good value for minimum area and threshold pair of parameters that should be used during testing to obtain good classification results. It relies on 10% of the defect-freee validation images and 20% of the defect and defect-free test images.

### Usage
usage: finetune.py [-h] -p  [-m] [-t]

optional arguments:

-h, --help      show this help message and exit

-p , --path     path to saved model

-m , --method   method for generating resmaps: 'ssim' or 'l2'

-t , --dtype    datatype for processing resmaps: 'float64' or 'uint8'


Example usage:
```
python3 finetune.py -p saved_models/mvtec/capsule/mvtec2/ssim/13-06-2020_15-35-10/CAE_mvtec2_b8_e39.hdf5 -m ssim -t float64
```

## Testing (`test.py`)

This script classifies test images using the threshold and the minimum defect area that have been previously determined by finetuning.

### Usage
usage: test.py [-h] -p

optional arguments:
  -h, --help    show this help message and exit
  -p , --path   path to saved model

Example usage:
```
python3 test.py -p saved_models/mvtec/capsule/mvtec2/ssim/13-06-2020_15-35-10/CAE_mvtec2_b8_e39.hdf5
```


Project Organization
------------

    ├── mvtec                       <- folder containing all mvtec classes.
    │   ├── bottle                  <- subfolder of a class (contains additional subfolders /train and /test).
    |   |── ...
    ├── autoencoder                 <- directory containing modules for training: autoencoder class and methods as well as custom losses and metrics.
    ├── processing                  <- directory containing modules for preprocessing images and before training and processing images after training.
    ├── results                     <- directory containing finetuning and test results.
    ├── readme.md                   <- readme file.
    ├── requirements.txt            <- requirement text file containing used libraries.
    ├── saved_models                <- directory containing saved models, training history, loss and learning plots and inspection images.
    ├── train.py                    <- training script to train the auto-encoder.
    ├── finetune.py                 <- approximates a good value for minimum area and threshold for classification.
    └── test.py                     <- test script to classify images of the test set using finetuned parameters.


--------

## Authors

* **Adnene Boumessouer** - (https://github.com/AdneneBoumessouer)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Paul Bergmann, the main author of the paper which this project relies on.
* François Chollet, author of the Keras deep learning library.
* Aurélien Géron, autor of the great book Hands on Machine Learning with Scikit-Learn, Keras and Tensorflow.
* Arun S. Maiya, author of the [ktrain](https://github.com/amaiya/ktrain) library: a wrapper for TensorFlow Keras that makes deep learning and AI more accessible and easier to apply.
* Adrian Rosebrock, author of the website [pyimagesearch](https://www.pyimagesearch.com/).
