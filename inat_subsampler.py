import os
import random
import shutil

random.seed(0)

path_to_train = './datasets/inat/train_mini/'
path_to_test = './datasets/inat/val/'

path_to_train_subset = './datasets/inat_mini/train_mini/'
path_to_test_subset = './datasets/inat_mini/val/'

def copy_random_subset(src_dir, dst_dir, subset_size):
    """
    Recursively copies a random subset of files from src_dir to dst_dir.
    """
    for root, dirs, files in os.walk(src_dir):
        # Create corresponding directories in the destination directory
        for dir in dirs:
            src_subdir = os.path.join(root, dir)
            dst_subdir = src_subdir.replace(src_dir, dst_dir)
            os.makedirs(dst_subdir, exist_ok=True)

        # Copy a random subset of files to the destination directory
        file_subset = random.sample(files, min(subset_size, len(files)))
        for file in file_subset:
            src_file = os.path.join(root, file)
            dst_file = src_file.replace(src_dir, dst_dir)
            shutil.copy2(src_file, dst_file)

copy_random_subset(path_to_train, path_to_train_subset, 5)
copy_random_subset(path_to_test, path_to_test_subset, 5)
