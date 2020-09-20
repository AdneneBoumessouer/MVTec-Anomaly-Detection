import os
import argparse
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from processing import utils
from processing import postprocessing
from processing.preprocessing import Preprocessor
from processing.preprocessing import get_preprocessing_function
from processing.postprocessing import label_images
from processing.utils import printProgressBar
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from test import predict_classes
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_largest_areas(resmaps, thresholds):

    # initialize largest areas to an empty list
    largest_areas = []

    # initialize progress bar
    printProgressBar(
        0, len(thresholds), prefix="Progress:", suffix="Complete", length=80
    )

    for index, threshold in enumerate(thresholds):
        # segment (threshold) residual maps
        resmaps_th = resmaps > threshold

        # compute labeled connected components
        _, areas_th = label_images(resmaps_th)

        # retieve largest area of all resmaps for current threshold
        areas_th_total = [item for sublist in areas_th for item in sublist]
        largest_area = np.amax(np.array(areas_th_total))
        largest_areas.append(largest_area)

        # print progress bar
        time.sleep(0.1)
        printProgressBar(
            index + 1, len(thresholds), prefix="Progress:", suffix="Complete", length=80
        )
    return largest_areas


def main(args):
    # Get validation arguments
    model_path = args.path
    method = args.method
    dtype = args.dtype

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

    # get the correct preprocessing function
    preprocessing_function = get_preprocessing_function(architecture)

    # ========= LOAD AND PREPROCESS VALIDATION & FINETUNING IMAGES =============

    # initialize preprocessor
    preprocessor = Preprocessor(
        input_directory=input_directory,
        rescale=rescale,
        shape=shape,
        color_mode=color_mode,
        preprocessing_function=preprocessing_function,
    )

    # -------------------------------------------------------------------

    # get validation generator
    validation_generator = preprocessor.get_val_generator(
        batch_size=nb_validation_images, shuffle=False
    )

    # retrieve preprocessed validation images from generator
    imgs_val_input = validation_generator.next()[0]

    # retrieve validation image_names
    filenames_val = validation_generator.filenames

    # reconstruct (i.e predict) validation images
    imgs_val_pred = model.predict(imgs_val_input)

    # instantiate TensorImages object to compute validation resmaps
    tensor_val = postprocessing.TensorImages(
        imgs_input=imgs_val_input,
        imgs_pred=imgs_val_pred,
        vmin=vmin,
        vmax=vmax,
        method=method,
        dtype=dtype,
        filenames=filenames_val,
    )

    # -------------------------------------------------------------------

    # get finetuning generator
    nb_test_images = preprocessor.get_total_number_test_images()

    finetuning_generator = preprocessor.get_finetuning_generator(
        batch_size=nb_test_images, shuffle=False
    )

    # retrieve preprocessed test images from generator
    imgs_test_input = finetuning_generator.next()[0]
    filenames_test = finetuning_generator.filenames

    # select a representative subset of test images for finetuning
    #  using stratified sampling
    assert "good" in finetuning_generator.class_indices
    index_array = finetuning_generator.index_array
    classes = finetuning_generator.classes
    _, index_array_ft, _, classes_ft = train_test_split(
        index_array,
        classes,
        test_size=config.FINETUNE_SPLIT,
        random_state=42,
        stratify=classes,
    )

    # get correct classes corresponding to selected images
    good_class_i = finetuning_generator.class_indices["good"]
    y_ft_true = np.array(
        [0 if class_i == good_class_i else 1 for class_i in classes_ft]
    )

    # select test images for finetuninig
    imgs_ft_input = imgs_test_input[index_array_ft]
    filenames_ft = list(np.array(filenames_test)[index_array_ft])

    # reconstruct (i.e predict) finetuning images
    imgs_ft_pred = model.predict(imgs_ft_input)

    # instantiate TensorImages object to compute finetuning resmaps
    tensor_ft = postprocessing.TensorImages(
        imgs_input=imgs_ft_input,
        imgs_pred=imgs_ft_pred,
        vmin=vmin,
        vmax=vmax,
        method=method,
        dtype=dtype,
        filenames=filenames_ft,
    )

    # ======================== COMPUTE THRESHOLDS ===========================

    # initialize finetuning dictionary
    dict_finetune = {
        "min_area": [],
        "threshold": [],
        "TPR": [],
        "TNR": [],
        "FPR": [],
        "FNR": [],
        "score": [],
    }

    # initialize discrete min_area values
    min_areas = np.arange(
        start=config.START_MIN_AREA,
        stop=config.STOP_MIN_AREA,
        step=config.STEP_MIN_AREA,
    )

    # initialize thresholds
    thresholds = np.arange(
        start=tensor_val.thresh_min,
        stop=tensor_val.thresh_max + tensor_val.thresh_step,
        step=tensor_val.thresh_step,
    )

    # compute largest anomaly areas in resmaps for increasing thresholds
    print("step 1/2: computing largest anomaly areas for increasing thresholds...")
    largest_areas = calculate_largest_areas(
        resmaps=tensor_val.resmaps, thresholds=thresholds,
    )

    # select best minimum area and threshold pair to use for testing
    print("step 2/2: selecting best minimum area and threshold pair for testing...")
    printProgressBar(
        0, len(min_areas), prefix="Progress:", suffix="Complete", length=80
    )

    for i, min_area in enumerate(min_areas):
        # compare current min_area with the largest area
        for index, largest_area in enumerate(largest_areas):
            if min_area > largest_area:
                break

        # select threshold corresponding to current min_area
        threshold = thresholds[index]

        # apply the min_area, threshold pair to finetuning images
        y_ft_pred = predict_classes(
            resmaps=tensor_ft.resmaps, min_area=min_area, threshold=threshold
        )

        # confusion matrix
        tnr, fpr, fnr, tpr = confusion_matrix(
            y_ft_true, y_ft_pred, normalize="true"
        ).ravel()

        # record current results
        dict_finetune["min_area"].append(min_area)
        dict_finetune["threshold"].append(threshold)
        dict_finetune["TPR"].append(tpr)
        dict_finetune["TNR"].append(tnr)
        dict_finetune["FPR"].append(fpr)
        dict_finetune["FNR"].append(fnr)
        dict_finetune["score"].append((tpr + tnr) / 2)

        # print progress bar
        printProgressBar(
            i + 1, len(min_areas), prefix="Progress:", suffix="Complete", length=80
        )

    # get min_area, threshold pair corresponding to best score
    max_score_i = np.argmax(dict_finetune["score"])
    max_score = float(dict_finetune["score"][max_score_i])
    best_min_area = int(dict_finetune["min_area"][max_score_i])
    best_threshold = float(dict_finetune["threshold"][max_score_i])

    # ===================== SAVE FINETUNING RESULTS ========================

    # create a results directory if not existent
    model_dir_name = os.path.basename(str(Path(model_path).parent))

    save_dir = os.path.join(
        os.getcwd(),
        "results",
        input_directory,
        architecture,
        loss,
        model_dir_name,
        "finetuning",
        "{}_{}".format(method, dtype),
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # save area and threshold pair
    finetuning_result = {
        "best_min_area": best_min_area,
        "best_threshold": best_threshold,
        "best_score": max_score,
        "method": method,
        "dtype": dtype,
        "split": config.FINETUNE_SPLIT,
    }
    print("finetuning results: {}".format(finetuning_result))

    # save validation result
    with open(os.path.join(save_dir, "finetuning_result.json"), "w") as json_file:
        json.dump(finetuning_result, json_file, indent=4, sort_keys=False)
    logger.info("finetuning results saved at {}".format(save_dir))

    # save finetuning plots
    plot_min_area_threshold(dict_finetune, index_best=max_score_i, save_dir=save_dir)
    plot_scores(dict_finetune, index_best=max_score_i, save_dir=save_dir)

    return


def plot_min_area_threshold(dict_finetune, index_best=None, save_dir=None):
    df_finetune = pd.DataFrame.from_dict(dict_finetune)
    with plt.style.context("seaborn-darkgrid"):
        df_finetune.plot(x="min_area", y=["threshold"], figsize=(12, 8))
        if index_best is not None:
            x = dict_finetune["min_area"][index_best]
            y = dict_finetune["threshold"][index_best]
            plt.axvline(x, 0, y, linestyle="dashed", color="red", linewidth=0.5)
            plt.axhline(y, 0, x, linestyle="dashed", color="red", linewidth=0.5)
            label_marker = "best min_area / threshold pair"
            plt.plot(x, y, markersize=10, marker="o", color="red", label=label_marker)
        title = "Min_Area Threshold plot\nbest min_area = {}\nbest threshold = {:.4f}".format(
            x, y
        )
        plt.title(title)
        plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "min_area_threshold_plot.png"))
        print("min_area threshold plot successfully saved at:\n {}".format(save_dir))
        plt.close()
    return


def plot_scores(dict_finetune, index_best=None, save_dir=None):
    df_finetune = pd.DataFrame.from_dict(dict_finetune)
    with plt.style.context("seaborn-darkgrid"):
        df_finetune.plot(x="min_area", y=["TPR", "TNR", "score"], figsize=(12, 8))
        if index_best is not None:
            x = dict_finetune["min_area"][index_best]
            y = dict_finetune["score"][index_best]
            plt.axvline(x, 0, 1, linestyle="dashed", color="red", linewidth=0.5)
            plt.plot(x, y, markersize=10, marker="o", color="red", label="best score")
        plt.title(f"Scores plot\nbest score = {y:.2E}")
        plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "scores_plot.png"))
        print("scores plot successfully saved at:\n {}".format(save_dir))
        plt.close()
    return


if __name__ == "__main__":

    # create parser
    parser = argparse.ArgumentParser(
        description="Determine good values for minimum area and threshold for classification."
    )
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
    )

    parser.add_argument(
        "-m",
        "--method",
        required=False,
        metavar="",
        choices=["ssim", "l2"],
        default="ssim",
        help="method for generating resmaps: 'ssim' or 'l2'",
    )

    parser.add_argument(
        "-t",
        "--dtype",
        required=False,
        metavar="",
        choices=["float64", "uint8"],
        default="float64",
        help="datatype for processing resmaps: 'float64' or 'uint8'",
    )

    args = parser.parse_args()

    main(args)

# Example of command to initiate finetuning with different resmap processing arguments (best combination: -m ssim -t float64)

# python3 finetune.py -p saved_models/mvtec/capsule/mvtecCAE/ssim/13-06-2020_15-35-10/mvtecCAE_b8_e39.hdf5 -m ssim -t float64
# python3 finetune.py -p saved_models/mvtec/capsule/mvtecCAE/ssim/13-06-2020_15-35-10/mvtecCAE_b8_e39.hdf5 -m ssim -t uint8
# python3 finetune.py -p saved_models/mvtec/capsule/mvtecCAE/ssim/13-06-2020_15-35-10/mvtecCAE_b8_e39.hdf5 -m l2 -t float64
# python3 finetune.py -p saved_models/mvtec/capsule/mvtecCAE/ssim/13-06-2020_15-35-10/mvtecCAE_b8_e39.hdf5 -m l2 -t uint8
