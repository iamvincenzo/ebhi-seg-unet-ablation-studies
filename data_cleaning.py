import os
from glob import glob


""" Helper function used to get the different 
    elements in the two lists. """
def diff(l1, l2):
    c = set(l1).union(set(l2))
    d = set(l1).intersection(set(l2))

    return list(c - d)

####################################### clean dataset - removing images without mask #######################################

""" Helper function used to remove images/masks without 
    the correspondings masks/images. """
def clean_dataset(args):
    classes = ['Normal', 'Polyp', 'Low-grade IN',
               'High-grade IN', 'Adenocarcinoma', 'Serrated adenoma']
    
    diffs = []

    print(f'\nPerforming dataset cleaning...\n')

    for c in classes:
        file_name = os.listdir(args.dataset_path + c + '/image')
        mask_name = os.listdir(args.dataset_path + c + '/label')

        print(f'{c}: {len(file_name)}, {len(mask_name)}')

        if len(diff(file_name, mask_name)) > 0:
            if (len(file_name) > len(mask_name)):
                # trace the directory where to remove files: 'image' or 'label'
                diffs.append((c, diff(file_name, mask_name), 'image'))
            else:
                diffs.append((c, diff(file_name, mask_name), 'label'))

    # debugging
    print(f'\nDifferences between image-label directories: {diffs}\n')
    
    # removing images without label(seg-mask) or mask without images
    removing_files = []
    for t in diffs:
        if len(t[1]) > 0:  # list that contains files name

            removing_files = [fn for n in t[1] for fn in glob(
                args.dataset_path + t[0] + '/' + t[2] + '/*') if n in fn]

            for rmf in set(removing_files):
                os.remove(rmf)

    print('Checking results: \n')

    for c in classes:
        file_name = os.listdir(args.dataset_path + c + '/image')
        mask_name = os.listdir(args.dataset_path + c + '/label')

        print(f'{c}: {len(file_name)}, {len(mask_name)}')

#######################################################################################################################


############################################# removing augmented images ###############################################

""" Helper function used to remove augmented images:
    images/masks generated with data_augmentation.py. """
def remove_aug(args):
    file_aug_list = [fn for fn in glob(
        args.dataset_path + '*/*/*') if 'aug' in fn]

    for filename in file_aug_list:
        os.remove(filename)

########################################################################################################################
