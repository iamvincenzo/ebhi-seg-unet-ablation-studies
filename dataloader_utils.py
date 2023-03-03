# Dataset creation functions
import math
import random
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def diff(l1, l2):
    """ This function is used to get the different
        elements in the two lists.

    Args:
        l1 (class 'list'): first list
        l2 (class 'list'): second list

    Returns:
        class 'list': list that contains all the elements that are in l1 but not in l2 
                      and all the elements that are in l2 but not in l1. 
    """
    c = set(l1).union(set(l2))
    d = set(l1).intersection(set(l2))

    return list(c - d)


def balanced_dataset(args):
    classes = ['Normal', 'Polyp', 'Low-grade IN',
               'High-grade IN', 'Adenocarcinoma', 'Serrated adenoma']

    """ Getting the number of elements per class. """
    lenghts = []
    for _, c in enumerate(classes):
        # the glob module finds all the pathnames matching a specified pattern
        tmp = glob(args.dataset_path + c + '/label/*')
        lenghts.append(len(tmp))
    print(f'\nNumber of elements per class: {lenghts}')

    """ Training-set should contain the 80% of the total samples:
        so we get 80% of elements for each class to create the train-set. 

        Test-set should contain the 20% of the toal samples:
        so we get 20% of elements for each class to create the test-set. 
        
        The total number of samples is 2226: 
            - 2226*0.8 = 1781 
            - 2226*0.2 = 445 
        
        The treshold th is used to get different round to obtain 1781
        as the number of samples contained in the train-set:
            ex:  int(math.ceil(0.8 * 61)) = 49 vs. int(0.8 * 61) = 48. 
    """
    th = args.th  # threshold used to get 2226*0.8=1781 (num-img in train-set)
    # list that contains the weighted number of elements per class to create the train-set
    w_train_clss = []
    for i in lenghts:
        if i < th:
            w_train_clss.append(int(math.ceil(0.8 * i)))
        else:
            w_train_clss.append(int(0.8 * i))
    w_test_clss = [e1 - e2 for (e1, e2) in zip(lenghts, w_train_clss)]
    print(f'Number of elements in train-set: {sum(w_train_clss)}')
    print(
        f'Number of elements in test-set: {sum(lenghts, 0) - sum(w_train_clss, 0)}')
    print(f'Number of elements per class to create train-set: {w_train_clss}')
    print(f'Number of elements per class to create test-set: {w_test_clss}')

    """ Idea of the following algorithm:
            1. Insert in mask_files_per_class all the paths of all images of all classes;
            2. Randomly select (without repetition) 80% of paths from mask_files_per_class 
                for each class and put them in mask_files_train;
            3. Use diff function to get the unseleted item and put them in mask_files_test.
    """
    random.seed(
        args.random_seed)  # hyperparameter used to create different random-sets

    mask_files_per_class = []
    mask_files_train = []
    img_files_train = []
    mask_files_test = []
    img_files_test = []

    for i, c in enumerate(classes):
        # creating a list of lists(path names per class)
        mask_files_per_class.append(
            glob(args.dataset_path + c + '/label/*'))

    for i, l in enumerate(mask_files_per_class):
        # generate n-unique samples from a sequence(list containing path names per class) without repetition.
        mylist = random.sample(l, w_train_clss[i])
        mask_files_train += mylist
        # insert in test-set non-selected-samples used to create training-set
        mask_files_test += diff(mylist, l)

    print(f'\nNumber of elements in mask_files_train: {len(mask_files_train)}')
    print(f'Number of elements in mask_files_test: {len(mask_files_test)}')

    for i in mask_files_train:
        # creating training-img path
        img_files_train.append(i.replace('label', 'image'))

    for i in mask_files_test:
        img_files_test.append(i.replace('label', 'image')
                              )  # creating test-img path

    print(f'Number of elements in img_files_train: {len(img_files_train)}')
    print(f'Number of elements in img_files_test: {len(img_files_test)}')

    # check for duplicates
    print(
        f'\nCheck duplicates in train/test set: {list(set(mask_files_train).intersection(mask_files_test))}\n')

    return img_files_train, mask_files_train, img_files_test, mask_files_test, w_train_clss, w_test_clss


class_dic = {
    'Normal': 0,
    'Polyp': 1,
    'Low-grade IN': 2,
    'High-grade IN': 3,
    'Adenocarcinoma': 4,
    'Serrated adenoma': 5
}


class EBHIDataset(Dataset):
    def __init__(self, image_paths, target_paths, args, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.args = args
        # random.seed(args.random_seed)  # hyperparameter used to create different random-sets ???

    def transform(self, image, mask):
        # data augmentation
        if self.args.apply_transformations == True:
            # random horizontal flipping (we apply transforms here because we need to apply
            # them with the same probability to both img and mask)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # random vertical flip
            if random.random() > 0.3:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            # random rotation
            if random.random() > 0.4:
                angle = random.randint(-30, 30)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        # to tensor and remove the alpha channel if present (PNG format)
        trnsf = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x[:3])])
        image = trnsf(image)
        mask = trnsf(mask)

        # input normalization
        if self.args.norm_input == True:
            image = TF.normalize(image, mean=(
                0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        return image, mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        x, y = self.transform(image, mask)

        # getting class label contained in the path-name
        res = [c for c in list(class_dic.keys())
               if c in self.image_paths[index]]
        l = class_dic[''.join(res)]

        return x, y, l

    def __len__(self):
        return len(self.image_paths)
