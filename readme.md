## NOTE: The work on this project is still in progress.

# Anomaly Datection

This project aims at developping a Deep Learning model using an unsupervided method to detect anomalies on images.

## Dataset

The dataset being used is the [MVTec dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

## Method

The method being used in this project is inspired to a great extent by the papers [MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf) and [Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders](https://arxiv.org/abs/1807.02011).
The method is devided in 3 steps: training, validating and testing.

## Prerequisites

### Dependencies
Run the following command to install required libraries and packages used in this project: 
```
pip install -r requirements.txt
```

### Download the Dataset
1. Download the mvtec dataset [here](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and save it to a directory of your choice (e.g in /Downloads)
2. Extract the compressed files.
3. Create a folder named **mvtec** in the project directory.
4. Move the extracted files (contaied in folders) to the **mvtec** folder.


### Directory Structure
For the scripts to work propoerly, it is required to have a specific directory structure. 
In the case of using the *mvtec* dataset, here is an example of how the directory stucture should look like:
------------

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

To train with your own dataset, you need to have a comparable directory structure. For example:
------------

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

During training, the CAE trains solely on defect-free images and learns to reconstruct defect-free training samples.

To initiate training, go to the project directory and run the following command in your terminal:
```
python3 train.py new -d <direcroty containing training images> -i <number of training instances> -a <architecture of the model to use> -b <batch size> -l <loss function> 
```
Here is an example:
```
python3 train.py new -d mvtec/capsule -i 10000 -a mvtec -b 24 -l mse 
```
**NOTE:** The number of training epochs will be determined by the given number of training images specified during the initiation of the training, devided by the actual number of images contained in the train folder of the chosen class.

## Validation (validate.py)

In this step, the threshold that will decide the classification in defect or defect-free images during testing will be determined using only a small portion of defect-free training images.

To initiate validation, go to the project directory and run the following command in your terminal:
```
python3 validate.py -p <path to trained model>
```
Here is an example:
```
python3 validate.py -p saved_models/MSE/17-02-2020_18:14:52/CAE_mvtec_b12.h5
```
**NOTE:** The method to determine the threshold is not yet finished. This script is still in progress.

## Testing (test.py)

During testing, the classification on the test images takes place using the threshold that was determined during validation. 

To initiate testing, go to the project directory and run the following command in your terminal:
```
python3 test.py -p <path to trained model>
```
Here is an example:
```
python3 test.py -p saved_models/MSE/17-02-2020_18:14:52/CAE_mvtec_b12.h5
```
**NOTE:** This script is still in progress and hasn't been tested yet.

## Project Organization

Project Organization
------------

    ├── mvtec                       <- folder containing all mvtec classes
    │   ├── bottle                  <- subfolder of a class (contains additional subfolders /train and /test)
    |   |── ...
    ├── custom_loss_functions.py    <- implments different loss functions to use in training
    ├── fine_tune.py                <- deprecated (to be remooved soon)
    ├── inspect_trained_model.ipynb <- notebook to inspect trained model
    ├── models.py                   <- contains different CAE architectures for training
    ├── readme.md                   <- readme file
    ├── requirements.txt            <- requirement text file containing all used packages and libraries
    ├── saved_models                <- directory containing saved models
    │   ├── L2                      <- saved models that trained with L2 loss
    │   ├── MSE                     <- saved models that trained with MSE loss
    │   ├── MSSIM                   <- saved models that trained with MSSIM loss
    │   └── SSIM                    <- saved models that trained with SSIM loss
    ├── test.py                     <- test script
    ├── train.py                    <- training script
    ├── utils.py                    <- utilitary and helper functions
    └── validate.py                 <- validation script


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


