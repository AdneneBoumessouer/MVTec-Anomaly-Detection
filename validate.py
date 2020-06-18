import os
import argparse
from pathlib import Path
import time
import json
import tensorflow as tf
from processing import utils
from processing import resmaps
from processing.preprocessing import Preprocessor
from processing.preprocessing import get_preprocessing_function
from processing.cv import label_images
from processing.utils import printProgressBar
from skimage.util import img_as_ubyte


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

    # =================== LOAD MODEL AND CONFIGURATION =========================

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

    # =========================== PREPROCESSING ===============================

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
    validation_generator = preprocessor.get_val_generator(
        batch_size=nb_validation_images, shuffle=False
    )

    # retrieve validation images from generator
    imgs_val_input = validation_generator.next()[0]

    # retrieve validation image_names
    filenames = validation_generator.filenames

    # get reconstructed images (i.e predictions) on validation dataset
    imgs_val_pred = model.predict(imgs_val_input)

    # convert to grayscale if RGB
    if color_mode == "rgb":
        imgs_val_input = tf.image.rgb_to_grayscale(imgs_val_input).numpy()
        imgs_val_pred = tf.image.rgb_to_grayscale(imgs_val_pred).numpy()

    # remove last channel since images are grayscale
    imgs_val_input = imgs_val_input[:, :, :, 0]
    imgs_val_pred = imgs_val_pred[:, :, :, 0]

    # ======================== COMPUTE THRESHOLD ===========================

    # instantiate TensorImages object
    tensor_val = resmaps.TensorImages(
        imgs_input=imgs_val_input,
        imgs_pred=imgs_val_pred,
        vmin=vmin,
        vmax=vmax,
        method=method,
        dtype=dtype,
        filenames=filenames,
    )

    # validation algorithm
    threshold = determine_threshold(
        resmaps=tensor_val.resmaps,
        min_area=min_area,
        thresh_min=tensor_val.thresh_min,
        thresh_max=tensor_val.thresh_max,
        thresh_step=tensor_val.thresh_step,
    )

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
