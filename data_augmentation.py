import cv2
import numpy as np
from glob import glob
from PIL import Image
import albumentations as A
from dataloader_utils import classes

""" Custom class used to: (i) apply some transformation to some images and masks 
    passed as argument; (ii) save them in order to augment data available for training 
    and testing (augment the entire dataset).  """
class AugmentData(object):
    """ Initialize configurations. """
    def __init__(self, args, image_paths, mask_paths):
        self.args = args
        self.image_paths = image_paths 
        self.mask_paths = mask_paths
        
        # invoke the method to augment the data
        self.augmentation()  

    """ Method used to augment image/mask data. """
    def augmentation(self):
        # some transformation applied to images and relative masks
        transform = A.Compose( 
            [
                A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                # RGBShift (perception of movement or color distortion)
                A.RGBShift(r_shift_limit=25, g_shift_limit=25,
                           b_shift_limit=25, p=0.9),
                A.OneOf([
                    # Blur (blur details in an image)
                    A.Blur(blur_limit=3, p=0.5),
                    # ColorJitter (modification of the following image parameters: 
                    # brightness, contrast, saturation, hue).
                    A.ColorJitter(p=0.5),
                ], p=1.0),
            ]
        )

        print('\nPerforming data augmentation...\n')

        for _, (img_path_label, path_msk) in enumerate(zip(self.image_paths, self.mask_paths)):
            path_img, label = img_path_label
            image = np.array(Image.open(path_img))
            mask = np.array(Image.open(path_msk))

            # foreach image/mask generates 'N=dataset_aug' images/masks
            for i in range(self.args.dataset_aug):
                augmentations = transform(image=image, mask=mask)
                augmented_img = augmentations['image']
                augmented_mask = augmentations['mask']

                sub_path_img = path_img.replace('.png', '')
                sub_path_msk = path_msk.replace('.png', '')

                # saving transformed image
                im = Image.fromarray(augmented_img)
                im.save(sub_path_img + '_aug_' + str(i + 1) + '_.png')

                # saving transformed mask
                im = Image.fromarray(augmented_mask)
                im.save(sub_path_msk + '_aug_' + str(i + 1) + '_.png')

    """ Method used to return paths of augmented images and masks. """
    def get_augmented_train_set(self):
        files = glob(self.args.dataset_path + '*/label/*')
        mask_files_train_aug_list = [fn for fn in files if 'aug' in fn]
        img_files_train_aug_list = []
        
        # insert the aumented images inside 'img_files_train_aug_list' using the
        # format used in 'dataloader_utils.py': 
        # (os.path.join(dataset_path, c, "image", e), i) 
        # = ('./data/EBHI-SEG/Normal\\image\\GT2016907-1-400-001.png', 0)
        for i, clss_name in enumerate(classes, 0): 
            img_files_train_aug_list += [(file_name.replace('label', 'image'), i) 
                                         for file_name in mask_files_train_aug_list 
                                         if clss_name in file_name]
        
        return img_files_train_aug_list, mask_files_train_aug_list
