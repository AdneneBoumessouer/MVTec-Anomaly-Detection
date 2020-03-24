import os
import sys
import argparse
from pathlib import Path
from modules import utils as utils
import json
import pandas as pd


def main(model_path, directory, architecture, loss):

    # get test directory
    model_dir_name = os.path.basename(str(Path(model_path).parent))
    test_dir = os.path.join(
        os.getcwd(), "results", directory, architecture, loss, model_dir_name, "test",
    )

    # list folders in test_dir
    subdirs = [x[0] for x in os.walk(test_dir)][1:]
    subdirs.sort()

    test_results_all = {"threshold": [], "min_area": [], "TPR": [], "TNR": []}

    for subdir in subdirs:
        with open(os.path.join(subdir, "test_results.json"), "r") as read_file:
            test_results = json.load(read_file)

        test_results_all["threshold"].append(test_results["threshold"])
        test_results_all["min_area"].append(test_results["min_area"])
        test_results_all["TPR"].append(test_results["TPR"])
        test_results_all["TNR"].append(test_results["TNR"])

    df_test_results_all = pd.DataFrame.from_dict(test_results_all)
    df_test_results_all.sort_values(by=["threshold", "min_area"])

    # save DataFrame
    with open(os.path.join(test_dir, "test_results_all.txt"), "a") as f:
        f.write(df_test_results_all.to_string(header=True, index=True))

    # print DataFrame to console
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_test_results_all)


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str, required=True, metavar="", help="path to saved model"
    )
    args = parser.parse_args()

    model_path = args.path

    # load setup
    setup = utils.get_model_setup(model_path)
    directory = setup["data_setup"]["directory"]
    architecture = setup["train_setup"]["architecture"]
    loss = setup["train_setup"]["loss"]

    main(model_path, directory, architecture, loss)

# python3 results.py -p saved_models/mvtec/capsule/mvtec2/MSE/25-02-2020_08-54-06/CAE_mvtec2_b12.h5
