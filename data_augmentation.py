import cv2
import numpy as np
from PIL import Image
import albumentations as A

""" Custom class used to: (i) apply some transformation
    to some images and masks passed as argument; 
    (ii) save them in order to augment data available 
    for training and testing (augment the entire dataset).  """
class AugmentData():
    def __init__(self, image_paths, mask_paths, args):
        super(AugmentData, self).__init__()
        self.image_paths = image_paths  # all the image paths of the dataset
        self.mask_paths = mask_paths  # all the corresponding mask paths
        self.args = args
        self.augmentation()  # invoke the method

    def augmentation(self):
        transform = A.Compose( # some transformation applied to images and relative masks
            [
                A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25,
                           b_shift_limit=25, p=0.9),
                A.OneOf([
                    A.Blur(blur_limit=3, p=0.5),
                    A.ColorJitter(p=0.5),
                ], p=1.0),
            ]
        )

        print('\nPerforming data augmentation...\n')

        for _, (path_img, path_msk) in enumerate(zip(self.image_paths, self.mask_paths)):
            image = np.array(Image.open(path_img))
            mask = np.array(Image.open(path_msk))

            # foreach image/mask generates "N = dataset_aug" images/masks
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
