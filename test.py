import sys
import os
import argparse
from pathlib import Path

from tensorflow import keras
from processing.preprocessing import Preprocessor
from processing.preprocessing import get_preprocessing_function
from processing.resmaps import calculate_resmaps as calculate_resmaps
from processing.cv import threshold_images as threshold_images
from processing.cv import label_images as label_images

from processing import utils as utils
from processing.utils import printProgressBar as printProgressBar

from sklearn.metrics import confusion_matrix
from skimage.util import img_as_ubyte

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json



def is_defective(areas, min_area):
    """Decides if image is defective given the areas of its connected components"""
    areas = np.array(areas)
    if areas[areas >= min_area].shape[0] > 0:
        return 1
    return 0


def classify(areas_all, min_area):
    """Decides if images are defective given the areas of their connected components"""
    y_pred = []
    for areas in areas_all:
        y_pred.append(is_defective(areas, min_area))
    return y_pred


def main(args):
    # parse arguments
    model_path = args.path
    save = args.save

    # ========================= LOAD SETUP ==============================

    # load model and info
    model, info, _ = utils.load_model_HDF5(model_path)

    input_directory = info["data"]["input_directory"]
    architecture = info["model"]["architecture"]
    loss = info["model"]["loss"]
    rescale = info["preprocessing"]["rescale"]
    shape = info["preprocessing"]["shape"]
    color_mode = info["model"]["color_mode"]
    nb_validation_images = info["data"]["nb_validation_images"]

    test_data_dir = os.path.join(input_directory, "train")

    # create directory to save test results
    model_dir_name = os.path.basename(str(Path(model_path).parent))

    save_dir = os.path.join(
        os.getcwd(),
        "results",
        input_directory,
        architecture,
        loss,
        model_dir_name,
        "test",
    )

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if args.adopt_validation:
        print("[INFO] adopting min_area and threshold yielded during validation...")
        # load area and corresponding threshold from validation json file
        parent_dir = str(Path(save_dir).parent)
        val_dir = os.path.join(parent_dir, "validation")
        with open(os.path.join(val_dir, "validation_result.json"), "r") as read_file:
            validation_result = json.load(read_file)

        min_area = validation_result["min_area"]
        threshold = validation_result["threshold"]

    else:
        print("[INFO] adopting user passed min_area and threshold...")
        threshold = args.threshold
        min_area = args.area

    # ============================= PREPROCESSING ===============================
    # get the correct preprocessing function
    preprocessing_function = get_preprocessing_function(architecture)

    # initialize preprocessor
    preprocessor = Preprocessor(
        input_directory=input_directory,
        rescale=rescale,
        shape=shape,
        color_mode=color_mode,
        preprocessing_function=preprocessing_function,
    )

    # get validation generator
    # nb_test_images = utils.get_total_number_test_images(test_data_dir)
    nb_test_images = preprocessor.get_total_number_test_images()
    test_generator = preprocessor.get_test_generator(
        batch_size=nb_test_images, shuffle=False
    )

    # retrieve validation images from generator
    imgs_test_input = test_generator.next()[0]

    # retrieve test image names
    filenames = test_generator.filenames

    # predict on test images
    imgs_test_pred = model.predict(imgs_test_input)

    # calculate residual maps on test set
    resmaps_test = calculate_resmaps(imgs_test_input, imgs_test_pred, method="SSIM")

    # Convert to 8-bit unsigned int
    resmaps_test = img_as_ubyte(resmaps_test)

    # ====================== Classification ==========================

    # initialize dictionary to store test results
    test_result = {"min_area": [], "threshold": [], "TPR": [], "TNR": []}

    # threshold residual maps with the given threshold
    resmaps_th = threshold_images(resmaps_test, threshold)[:, :, :, 0]

    # compute connected components
    resmaps_labeled, areas_all = label_images(resmaps_th)

    # classify images
    y_pred = classify(areas_all, min_area)

    # retrieve ground truth
    y_true = [1 if "good" not in filename.split("/") else 0 for filename in filenames]

    # condition positive (P)
    P = y_true.count(1)

    # condition negative (N)
    N = y_true.count(0)

    # true positive (TP)
    TP = np.sum(
        [1 if y_pred[i] == y_true[i] == 1 else 0 for i in range(nb_test_images)]
    )

    # true negative (TN)
    TN = np.sum(
        [1 if y_pred[i] == y_true[i] == 0 else 0 for i in range(nb_test_images)]
    )

    # sensitivity, recall, hit rate, or true positive rate (TPR)
    TPR = TP / P

    # specificity, selectivity or true negative rate (TNR)
    TNR = TN / N

    # confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, normalize="true").ravel()

    # append test results to dictionary
    test_result["min_area"].append(min_area)
    test_result["threshold"].append(threshold)
    test_result["TPR"].append(TPR)
    test_result["TNR"].append(TNR)

    # save validation result
    with open(os.path.join(save_dir, "test_result.json"), "w") as json_file:
        json.dump(test_result, json_file, indent=4, sort_keys=False)

    # save classification of image files in a .txt file
    classification = {
        "filenames": filenames,
        "predictions": y_pred,
        "truth": y_true,
        "accurate_predictions": np.array(y_true) == np.array(y_pred),
    }
    df_clf = pd.DataFrame.from_dict(classification)
    with open(os.path.join(save_dir, "classification.txt"), "w") as f:
        f.write("min_area = {}, threshold = {}\n\n".format(min_area, threshold))
        f.write(df_clf.to_string(header=True, index=True))

    # print classification results to console
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_clf)

    # print test_results to console
    print("[INFO] test results: {}".format(test_result))

    if save:
        utils.save_np(imgs_val_input, save_dir, "imgs_test_input.npy")
        utils.save_np(imgs_val_pred, save_dir, "imgs_test_pred.npy")
        utils.save_np(resmaps_val, save_dir, "resmaps_test.npy")


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
    )

    parser.add_argument(
        "--adopt-validation",
        action="store_true",
        help="whether or not to (min_area, threshold) value pairs during validation",
    )

    parser.add_argument(
        "-t", "--threshold", type=int, default=220, metavar="", help="threshold(s)",
    )

    parser.add_argument(
        "-a", "--area", type=int, default=80, metavar="", help="area(s)",
    )

    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="save inputs, predictions and reconstructions of validation dataset",
    )

    args = parser.parse_args()

    main(args)

# Examples of command to initiate testing

# using threshold and area from validation results:
# python3 test.py -p saved_models/MSE/25-02-2020_08-54-06/CAE_mvtec2_b12.h5

# using passed arguments for threshold and area
# python3 test.py -p saved_models/mvtec/capsule/mvtec2/SSIM/19-04-2020_14-14-36/CAE_mvtec2_b8.h5 --adopt-validation

# python3 test.py -p saved_models/mvtec/capsule/mvtec2/SSIM/19-04-2020_14-14-36/CAE_mvtec2_b8.h5 -a 10 -t 200
