import os
import argparse
from pathlib import Path
import time
import json
import numpy as np
import tensorflow as tf
from processing import utils
from processing import resmaps
from processing.preprocessing import Preprocessor
from processing.preprocessing import get_preprocessing_function
from processing.cv import label_images
from processing.utils import printProgressBar
from skimage.util import img_as_ubyte

from sklearn.model_selection import train_test_split

FINETUNE_SPLIT = 0.2


def determine_threshold(resmaps, min_area, thresh_min, thresh_max, thresh_step):
    # set initial threshold, counter and max number of steps
    threshold = thresh_min
    index = 0
    n_steps = (thresh_max - thresh_min) // thresh_step

    # initialize progress bar
    printProgressBar(0, n_steps, prefix="Progress:", suffix="Complete", length=50)

    while True:
        # segment (threshold) residual maps
        resmaps_th = resmaps > threshold
        # compute labeled connected components
        resmaps_labeled, areas_all = label_images(resmaps_th)
        # check if area of largest anomalous region is below the minimum area
        areas_all_flat = [item for sublist in areas_all for item in sublist]
        areas_all_flat.sort(reverse=True)
        try:
            if min_area > areas_all_flat[0]:
                time.sleep(0.1)
                printProgressBar(
                    n_steps, n_steps, prefix="Progress:", suffix="Complete", length=50
                )
                break
        except IndexError:
            continue

        if threshold > thresh_max:
            break

        threshold = threshold + thresh_step
        index = index + 1

        # print progress bar
        time.sleep(0.1)
        printProgressBar(
            index, n_steps, prefix="Progress:", suffix="Complete", length=50
        )
    return threshold


def main(args):
    # Get validation arguments
    model_path = args.path
    method = args.method
    dtype = args.dtype
    min_area = args.area
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

    # ==================== PREPROCESS VALIDATION IMAGES ========================

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

    # -------------------------------------------------------------------

    # get validation generator
    validation_generator = preprocessor.get_val_generator(
        batch_size=nb_validation_images, shuffle=False
    )

    # retrieve validation images from generator
    imgs_val_input = validation_generator.next()[0]

    # retrieve validation image_names
    filenames_val = validation_generator.filenames_val

    # get reconstructed images (i.e predictions) on validation dataset
    imgs_val_pred = model.predict(imgs_val_input)

    # convert to grayscale if RGB
    if color_mode == "rgb":
        imgs_val_input = tf.image.rgb_to_grayscale(imgs_val_input).numpy()
        imgs_val_pred = tf.image.rgb_to_grayscale(imgs_val_pred).numpy()

    # remove last channel since images are grayscale
    imgs_val_input = imgs_val_input[:, :, :, 0]
    imgs_val_pred = imgs_val_pred[:, :, :, 0]

    # instantiate TensorImages object to compute validation resmaps
    tensor_val = resmaps.TensorImages(
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
    # nb_finetuning_images = int(FINETUNE_SPLIT * nb_test_images)  # CHANGE

    finetuning_generator = preprocessor.get_finetuning_generator(
        batch_size=nb_test_images, shuffle=False
    )

    assert "good" in finetuning_generator.class_indices

    # select a representative subset of test images for finetuning (stratified sampling)
    index_array = finetuning_generator.index_array
    classes = finetuning_generator.classes
    _, index_array_ft, _, classes_ft = train_test_split(
        index_array,
        classes,
        test_size=FINETUNE_SPLIT,
        random_state=42,
        stratify=classes,
    )

    # get correct classes corresponding to selected images
    good_class_i = finetuning_generator.class_indices["good"]
    y_ft_true = np.array(
        [0 if class_i == good_class_i else 1 for class_i in classes_ft]
    )

    # select test images for finetuninig
    imgs_ft_input = finetuning_generator.next()[0][index_array_ft]
    # imgs_ft_input = imgs_ft_input[index_array_ft]

    # get reconstructed images (i.e predictions) on validation dataset
    imgs_ft_pred = model.predict(imgs_ft_input)

    # convert to grayscale if RGB
    if color_mode == "rgb":
        imgs_ft_input = tf.image.rgb_to_grayscale(imgs_ft_input).numpy()
        imgs_ft_pred = tf.image.rgb_to_grayscale(imgs_ft_pred).numpy()

    # remove last channel since images are grayscale
    imgs_ft_input = imgs_ft_input[:, :, :, 0]
    imgs_ft_pred = imgs_ft_pred[:, :, :, 0]

    # instantiate TensorImages object to compute finetuning resmaps
    tensor_ft = resmaps.TensorImages(
        imgs_input=imgs_ft_input,
        imgs_pred=imgs_ft_pred,
        vmin=vmin,
        vmax=vmax,
        method=method,
        dtype=dtype,
        filenames=filenames,
    )

    # ======================== COMPUTE THRESHOLD ===========================

    # initialize finetuning dictionary
    dict_finetune = {"min_area": [], "threshold": [], "TPR": [], "TNR": [], "score": []}

    # create discrete min_area values
    min_areas = np.arange(start=5, stop=505, step=5)
    length = len(min_areas)

    for i, min_area in enumerate(min_areas):
        # compute threshold corresponding to current min_area
        threshold = determine_threshold(
            resmaps=tensor_val.resmaps,
            min_area=min_area,
            thresh_min=tensor_val.thresh_min,
            thresh_max=tensor_val.thresh_max,
            thresh_step=tensor_val.thresh_step,
        )

        # apply the min_area, threshold pair to finetuning images
        y_ft_pred = predict_classes(
            resmaps=tensor_ft.resmaps, min_area=min_area, threshold=threshold
        )

        # confusion matrix
        tnr, fp, fn, tpr = confusion_matrix(y_true, y_pred, normalize="true").ravel()

        # record finetuning results
        dict_finetune["min_area"].append(min_area)
        dict_finetune["threshold"].append(threshold)
        dict_finetune["TPR"].append(tpr)
        dict_finetune["TNR"].append(tnr)
        dict_finetune["score"].append(np.mean(tpr, tnr))

    # get min_area, threshold pair corresponding to best score
    max_score_i = np.argmax(dict_finetune["score"])
    best_min_area = dict_finetune["min_area"][max_score_i]
    best_threshold = dict_finetune["threshold"][max_score_i]

    # ===================== SAVE VALIDATION RESULTS ========================

    # create a results directory if not existent
    model_dir_name = os.path.basename(str(Path(model_path).parent))

    save_dir = os.path.join(
        os.getcwd(),
        "results",
        input_directory,
        architecture,
        loss,
        model_dir_name,
        "validation",
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # save area and threshold pair
    validation_result = {
        "min_area": min_area,
        "threshold": threshold,
        "method": method,
        "dtype": dtype,
    }
    print("[INFO] validation results: {}".format(validation_result))

    # save validation result
    with open(os.path.join(save_dir, "validation_result.json"), "w") as json_file:
        json.dump(validation_result, json_file, indent=4, sort_keys=False)
    print("[INFO] validation results saved at {}".format(save_dir))

    if save:
        utils.save_np(imgs_val_input, save_dir, "imgs_val_input.npy")
        utils.save_np(imgs_val_pred, save_dir, "imgs_val_pred.npy")
        utils.save_np(resmaps_val, save_dir, "resmaps_val.npy")


if __name__ == "__main__":

    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
    )

    parser.add_argument(
        "-a",
        "--area",
        type=int,
        required=True,
        metavar="",
        help="minimum area for a connected component to be classified as anomalous",
    )

    parser.add_argument(
        "-m",
        "--method",
        required=False,
        choices=["SSIM", "L2"],
        default="SSIM",
        help="method for computing resmaps",
    )

    parser.add_argument(
        "-t",
        "--dtype",
        required=False,
        choices=["float64", "uint8"],
        default="float64",
        help="datatype for image processing: float64 or uint8",
    )

    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="save inputs, predictions and reconstructions of validation dataset",
    )

    args = parser.parse_args()

    main(args)

# Example of command to initiate validation with different resmap processing arguments (best combination: -m SSIM -t float64)

# python3 validate.py -p saved_models/mvtec/capsule/mvtec2/ssim/13-06-2020_15-35-10/CAE_mvtec2_b8_e39.hdf5 -a 150 -m SSIM -t float64
# python3 validate.py -p saved_models/mvtec/capsule/mvtec2/ssim/13-06-2020_15-35-10/CAE_mvtec2_b8_e39.hdf5 -a 150 -m SSIM -t uint8
# python3 validate.py -p saved_models/mvtec/capsule/mvtec2/ssim/13-06-2020_15-35-10/CAE_mvtec2_b8_e39.hdf5 -a 150 -m L2 -t float64
# python3 validate.py -p saved_models/mvtec/capsule/mvtec2/ssim/13-06-2020_15-35-10/CAE_mvtec2_b8_e39.hdf5 -a 150 -m L2 -t uint8
