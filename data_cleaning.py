import os
from glob import glob
from dataloader_utils import classes


""" Helper function used to remove images 
    without their corresponding masks and viceversa.  """
def clean_dataset(args):
    print(f'\nPerforming dataset cleaning...\n')

    for c in classes:
        imgs_path = os.path.join(args.dataset_path, c, 'image')
        masks_path = os.path.join(args.dataset_path, c, 'label')

        # convert the list to a set in order to perform operations such
        # as identifying elements that are in one list but not in the other
        imgs = set(os.listdir(imgs_path))
        masks = set(os.listdir(masks_path))
        # return the files in imgs without
        # corresponding mask (need to be removed)
        missing_masks = imgs - masks
        # return the files in masks without
        # corresponding images (need to be removed)
        missing_images = masks - imgs

        print(f"Class analysis: {c}...")

        # removing images without corresponding masks
        for image_file in missing_masks:
            os.remove(os.path.join(imgs_path, image_file))
            print(f'\tRemoved image file: {image_file}')

        # removoving masks without corresponding images
        for label_file in missing_images:
            os.remove(os.path.join(masks_path, label_file))
            print(f'\tRemoved mask file: {label_file}')

        print(f"{c}: {len(set(os.listdir(imgs_path)))}, "
              f" {len(set(os.listdir(masks_path)))}\n")

    print('Data cleaning done...\n')

""" Helper function used to remove augmented images:
    (images, masks) generated with 'data_augmentation.py'. """
def remove_aug(args):
    file_aug_list = [fn for fn in glob(
        args.dataset_path + '*/*/*') if 'aug' in fn]

    for filename in file_aug_list:
        os.remove(filename)
