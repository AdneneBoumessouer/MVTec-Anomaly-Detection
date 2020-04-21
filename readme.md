## NOTE: The work on this project is still in progress.

# Anomaly Detection

This project aims at developping a Deep Learning model using an unsupervided method to detect surface anomalies on images.

## Dataset

The dataset being used is the [MVTec dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/).

## Method

The method being used in this project is inspired to a great extent by the papers [MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf) and [Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders](https://arxiv.org/abs/1807.02011).
The method is devided in 3 steps: training, validating and testing.

## Prerequisites

### Dependencies
Libraries and packages used in this project: 
* `tensorflow-gpu 2.1.0`
* `Keras 2.3.1`
* `ktrain 0.13.0`
* `scikit-image 0.16.2`
* `opencv-python 4.2.0.34`
* `pandas 1.0.3`
* `numpy 1.18.1`
* `matplotlib 3.1.3`


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

## Training (`train.py`)

The method uses a Convolutional Auto-Encoder (CAE). There are two proposed variants:
* CAE proposed in the paper [MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf)
* CAE that uses Keras's inception_resnet_v2 CNN-model pretrained on imagenet as the Encoder. 
The Decoder is inspired by the paper [Anomaly Detection and Localization in Images using Guided Attention](https://openreview.net/forum?id=B1gikpEtwH)

During training, the CAE trains exclusively on defect-free images and learns to reconstruct defect-free training samples.

To initiate training, go to the project directory and run the following command in your terminal:
```
python3 train.py -d <direcroty containing training images> -i <number of training instances> -a <architecture of the model to use> -b <batch size> -l <loss function> -c <color mode> -t <tag>
```
Here is an example:
```
python3 train.py -d mvtec/capsule -a mvtec -b 12 -l ssim -c grayscale -t "first try"
```
**NOTE:** The number of training epochs will be determined by the given argument `<number of training instances>` specified during the initiation of the training, devided by the actual number of images contained in the train folder of the chosen class.
Example: if you passed `-i 10000` and the training directory contains `1000` images, then the number of epochs will be equal to `10`.

## Finetuning (`finetune.ipynb`)

This script aims at guiding the the user in chosing the right minimum defect area size that an anomalous region in the thresholded Resmaps must have to be classified as defective. This value will be used during the validation step by `validate.py`.
It investigates the following features:
* Number of anomalous regions with increasing thresholds.
* Sum, Mean and Standard Deviation of the anomalous regions' area size with increasing thresholds.
* Distribution of the the anomalous regions' area size with validation ResMaps for a range of thresholds given bu the user.


## Validation (`validate.py`)

This script computes the threshold that must be used in classification during testing. It uses a small portion (10%) of defect-free training images (validation set). To determine the threshold, a minimum defect area must be passed as an argument. This value is determined in the finetuning step using the Jupyter-Notebook `finetune.ipynb`.

To initiate validation, go to the project directory and run the following command in your terminal:
```
python3 validate.py -p <path to trained model> -a <minimum defective area>
```
Here is an example:
```
python3 validate.py -p saved_models/mvtec/capsule/mvtec2/SSIM/19-04-2020_14-14-36/CAE_mvtec2_b8.h5 -a 10
```

## Testing (`test.py`)

During testing, the classification on the test images takes place using two input parameters, a threshold and a minimum defect area. You can either use the optimal (minimum area, threshold) pair that have been previously determined by the finetuning and validation step respectively by passing the flag `--adopt-validation` or pass your own pair of area and threshold values.

To initiate testing using the values gained from the validation step, go to the project directory and run the following command in your terminal:
```
python3 test.py -p <path to trained model> --adopt-validation
```
Example:
```
python3 test.py -p saved_models/mvtec/capsule/mvtec2/SSIM/19-04-2020_14-14-36/CAE_mvtec2_b8.h5 --adopt-validation
```
To use your own pair of minimum area and threshold, use:
```
python3 test.py -p <path to trained model> -t <threshold> -a <area>
```
Example:
```
python3 test.py -p saved_models/mvtec/capsule/mvtec2/SSIM/19-04-2020_14-14-36/CAE_mvtec2_b8.h5 -t 28 -a 50
```

Project Organization
------------

    ├── mvtec                       <- folder containing all mvtec classes.
    │   ├── bottle                  <- subfolder of a class (contains additional subfolders /train and /test).
    |   |── ...
    ├── modules                     <- module containing model definitions, submodules for image processing and helper functions.
    ├── results                     <- directory containing validation and results.
    ├── readme.md                   <- readme file.
    ├── requirements.txt            <- requirement text file containing used libraries.
    ├── saved_models                <- directory containing saved models, training history, loss and learning plots and inspection images.
    ├── train.py                    <- training script to train the auto-encoder.
    ├── finetune.ipynb              <- Jupyter-Notebook to approximate minimum area.
    ├── validate.py                 <- validation script to determine classification threshold.
    └── test.py                     <- test script to classify images of the test set.


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
* Arun S. Maiya, author of [ktrain](https://github.com/amaiya/ktrain)
* Adrian Rosebrock, author of the website [pyimagesearch](https://www.pyimagesearch.com/)
