import sys
import os
import argparse
from pathlib import Path
import time
import json
import tensorflow as tf
from processing import utils
from processing import postprocessing
from processing.preprocessing import Preprocessor
from processing.preprocessing import get_preprocessing_function
from processing.postprocessing import label_images
from processing.utils import printProgressBar
from skimage.util import img_as_ubyte
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_true_classes(filenames):
    # retrieve ground truth
    y_true = [1 if "good" not in filename.split("/") else 0 for filename in filenames]
    return y_true


def is_defective(areas, min_area):
    """Decides if image is defective given the areas of its connected components"""
    areas = np.array(areas)
    if areas[areas >= min_area].shape[0] > 0:
        return 1
    return 0


def predict_classes(resmaps, min_area, threshold):
    # threshold residual maps with the given threshold
    resmaps_th = resmaps > threshold
    # compute connected components
    _, areas_all = label_images(resmaps_th)
    # Decides if images are defective given the areas of their connected components
    y_pred = [is_defective(areas, min_area) for areas in areas_all]
    return y_pred


def save_segmented_images(resmaps, threshold, filenames, save_dir):
    # threshold residual maps with the given threshold
    resmaps_th = resmaps > threshold
    # create directory to save segmented resmaps
    seg_dir = os.path.join(save_dir, "segmentation")
    if not os.path.isdir(seg_dir):
        os.makedirs(seg_dir)
    # save segmented resmaps
    for i, resmap_th in enumerate(resmaps_th):
        fname = utils.generate_new_name(filenames[i], suffix="seg")
        fpath = os.path.join(seg_dir, fname)
        plt.imsave(fpath, resmap_th, cmap="gray")
    return


def main(args):
    # parse arguments
    model_path = args.path
    save = args.save

    # ============= LOAD MODEL AND PREPROCESSING CONFIGURATION ================

    # load model and info
    model, info, _ = utils.load_model_HDF5(model_path)
    # set parameters
    input_directory = info["data"]["input_directory"]
    architecture = info["model"]["architecture"]
    loss = info["model"]["loss"]
    rescale = info["preprocessing"]["rescale"]
    shape = info["preprocessing"]["shape"]
    color_mode = info["preprocessing"]["color_mode"]
    vmin = info["preprocessing"]["vmin"]
    vmax = info["preprocessing"]["vmax"]
    nb_validation_images = info["data"]["nb_validation_images"]

    # =================== LOAD VALIDATION PARAMETERS =========================

    model_dir_name = os.path.basename(str(Path(model_path).parent))
    finetune_dir = os.path.join(
        os.getcwd(),
        "results",
        input_directory,
        architecture,
        loss,
        model_dir_name,
        "finetuning",
    )
    subdirs = os.listdir(finetune_dir)
    for subdir in subdirs:
        logger.info(
            "testing with finetuning parameters from \n{}...".format(
                os.path.join(finetune_dir, subdir)
            )
        )
        try:
            with open(
                os.path.join(finetune_dir, subdir, "finetuning_result.json"), "r"
            ) as read_file:
                validation_result = json.load(read_file)
        except FileNotFoundError:
            logger.warning("run finetune.py before testing.\nexiting script.")
            sys.exit()

        min_area = validation_result["best_min_area"]
        threshold = validation_result["best_threshold"]
        method = validation_result["method"]
        dtype = validation_result["dtype"]

        # ====================== PREPROCESS TEST IMAGES ==========================

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

        # get test generator
        nb_test_images = preprocessor.get_total_number_test_images()
        test_generator = preprocessor.get_test_generator(
            batch_size=nb_test_images, shuffle=False
        )

        # retrieve test images from generator
        imgs_test_input = test_generator.next()[0]

        # retrieve test image names
        filenames = test_generator.filenames

        # predict on test images
        imgs_test_pred = model.predict(imgs_test_input)

        # instantiate TensorImages object
        tensor_test = postprocessing.TensorImages(
            imgs_input=imgs_test_input,
            imgs_pred=imgs_test_pred,
            vmin=vmin,
            vmax=vmax,
            method=method,
            dtype=dtype,
            filenames=filenames,
        )

        # ====================== CLASSIFICATION ==========================

        # retrieve ground truth
        y_true = get_true_classes(filenames)

        # predict classes on test images
        y_pred = predict_classes(
            resmaps=tensor_test.resmaps, min_area=min_area, threshold=threshold
        )

        # confusion matrix
        tnr, fp, fn, tpr = confusion_matrix(y_true, y_pred, normalize="true").ravel()

        # initialize dictionary to store test results
        test_result = {
            "min_area": min_area,
            "threshold": threshold,
            "TPR": tpr,
            "TNR": tnr,
            "score": (tpr + tnr) / 2,
            "method": method,
            "dtype": dtype,
        }

        # ====================== SAVE TEST RESULTS =========================

        # create directory to save test results
        save_dir = os.path.join(
            os.getcwd(),
            "results",
            input_directory,
            architecture,
            loss,
            model_dir_name,
            "test",
            subdir,
        )

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # save test result
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
            f.write(
                "min_area = {}, threshold = {}, method = {}, dtype = {}\n\n".format(
                    min_area, threshold, method, dtype
                )
            )
            f.write(df_clf.to_string(header=True, index=True))

        # print classification results to console
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df_clf)

        # save segmented resmaps
        if save:
            save_segmented_images(tensor_test.resmaps, threshold, filenames, save_dir)

        # print test_results to console
        print("test results: {}".format(test_result))


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
    )
    parser.add_argument(
        "-s", "--save", action="store_true", help="save segmented images",
    )

    args = parser.parse_args()

    main(args)

# Examples of command to initiate testing
# python3 test.py -p saved_models/mvtec/capsule/mvtecCAE/ssim/13-06-2020_15-35-10/mvtecCAE_b8_e39.hdf5
