import os
import shutil

"""
Pre-requisite to use this function:
Your current working directory must contain a directory named "mvtec" containing the mvtec dataset!
This script creates following directory structure necessary for Keras's ImageDataGenerator class.

            

For more information on directory structure, see example at:
https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
https://realpython.com/working-with-files-in-python/#getting-a-directory-listing         
"""


def copy_and_rename_file(src_dir, old_filename, dst_dir, new_filename):
    """Copies file in a new directory with a new name"""
    src_file = os.path.join(src_dir, old_filename)
    shutil.copy(src_file, dst_dir)
    dst_file = os.path.join(dst_dir, old_filename)
    new_dst_file = os.path.join(dst_dir, new_filename)
    os.rename(dst_file, new_dst_file)


def copy_and_rename_mvtec_train(mvtec_dir, dst_train_dir):
    """Copies and renames all MVTec training images to a single new directory.
    This is necessary for the use of Keras's ImageDataGenerator Class"""
    class_names = next(os.walk(mvtec_dir))[1]
    class_names.sort()
    for class_name in class_names:
        src_dir = os.path.join(mvtec_dir, class_name, "train/good")
        # get image names
        filenames = [
            name
            for name in os.listdir(src_dir)
            if os.path.isfile(os.path.join(src_dir, name))
        ]
        filenames.sort()
        for old_filename in filenames:
            new_filename = class_name + "_" + old_filename
            copy_and_rename_file(src_dir, old_filename, dst_train_dir, new_filename)


def copy_and_rename_mvtec_train_validation(
    mvtec_dir, dst_train_dir, dst_valid_dir, validation_split=0.1
):
    """Copies and renames all MVTec training images to two separate directories: one for training and one for validating.
    This directory structure is necessary for the use of Keras's ImageDataGenerator classmethod flow_from_directory()"""
    
    # create target directories
    os.makedirs(dst_train_dir, mode=0o770, exist_ok=True)
    os.makedirs(dst_valid_dir, mode=0o770, exist_ok=True)

    class_names = next(os.walk(mvtec_dir))[1]
    class_names.sort()
    for class_name in class_names:
        src_dir = os.path.join(mvtec_dir, class_name, "train/good")
        # get image names
        filenames = [
            name
            for name in os.listdir(src_dir)
            if os.path.isfile(os.path.join(src_dir, name))
        ]
        filenames.sort()
        # compute split index
        split_index = int(len(filenames) * (1 - validation_split))
        for old_filename in filenames[:split_index]:
            new_filename = class_name + "_" + old_filename
            copy_and_rename_file(src_dir, old_filename, dst_train_dir, new_filename)

        for old_filename in filenames[split_index:]:
            new_filename = class_name + "_" + old_filename
            copy_and_rename_file(src_dir, old_filename, dst_valid_dir, new_filename)


if __name__ == "__main__":    
    copy_and_rename_mvtec_train_validation(
        mvtec_dir="mvtec",
        dst_train_dir="datasets/data/train/good",
        dst_valid_dir="datasets/data/validation/good",
        validation_split=0.1,
    )

