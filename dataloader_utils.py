import os
import math
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class_dic = {
    'Normal': 0,
    'Polyp': 1,
    'Low-grade IN': 2,
    'High-grade IN': 3,
    'Adenocarcinoma': 4,
    'Serrated adenoma': 5
}

classes = ['Normal', 'Polyp', 'Low-grade IN',
           'High-grade IN', 'Adenocarcinoma', 'Serrated adenoma']


""" Helper function used to get the number of elements per class in the dataset."""
def get_num_elements_per_class(dataset_path, classes):
    lengths = [len(os.listdir(os.path.join(dataset_path, c, "image"))) for c in classes]
    return lengths

""" Function used to generate a proportioned dataset: 80% training and 
    20% test(validation) per each class. """
def get_proportioned_dataset(dataset_path, classes, th, random_seed):
    lengths = get_num_elements_per_class(dataset_path, classes)
    print(f"\nNumber of elements per class: {lengths}")

    # calculate number of elements for training and test sets
    num_elements = sum(lengths)
    num_train_elements = int(math.ceil(0.8 * num_elements))
    num_test_elements = num_elements - num_train_elements
    print(f"Number of elements in train set: {num_train_elements}")
    print(f"Number of elements in test set: {num_test_elements}")

    # calculate number of elements per class in training and test sets
    train_lengths = [int(math.ceil(0.8 * l)) if l < th else int(0.8 * l) for l in lengths]
    test_lengths = [l - t for l, t in zip(lengths, train_lengths)]
    print(f"Number of elements per class in train set: {train_lengths}")
    print(f"Number of elements per class in test set: {test_lengths}")

    # generate train and test sets
    random.seed(random_seed)
    mask_files_train = []
    img_files_train = []
    mask_files_test = []
    img_files_test = []

    """ Idea of the following algorithm:
        for each class:
            1. Instantiate the class_elements list with all the paths of the images belonging 
                to a specific class under analysis;
            2. Randomly select (without repetition) 80% of paths from class_elements and put 
                them in train_elements (with the corresponding label);
            3. Using 'set' function to get the unselected item of class_elements and put them 
                in test_set (with the corresponding label).
    """
    for i, c in enumerate(classes, 0):
        # get list of all elements in class
        class_elements = os.listdir(os.path.join(dataset_path, c, "image"))

        # randomly select elements for train and test sets
        train_elements = random.sample(class_elements, train_lengths[i])
        test_elements = list(set(class_elements) - set(train_elements))

        # add elements to train and test sets
        # (os.path.join(dataset_path, c, "image", e), i) 
        # = ('./data/EBHI-SEG/Normal\\image\\GT2016907-1-400-001.png', 0)
        img_files_train += [(os.path.join(dataset_path, c, "image", e), i) for e in train_elements]
        img_files_test += [(os.path.join(dataset_path, c, "image", e), i) for e in test_elements]

    mask_files_train = [file[0].replace('image', 'label') for file in img_files_train]
    mask_files_test = [file[0].replace('image', 'label') for file in img_files_test]

    print(f'Number of elements in img_files_train: {len(img_files_train)}')
    print(f'Number of elements in img_files_test: {len(img_files_test)}')

    # # debug: check for duplicates
    # print(f'\nCheck duplicates in train/test set: {list(set(mask_files_train).intersection(mask_files_test))}\n')

    return img_files_train, mask_files_train, img_files_test, mask_files_test, train_lengths, test_lengths


""" Custom class used to create the training and test sets. """
class EBHIDataset(Dataset):
    """ Initialize configurations. """
    def __init__(self, image_paths, target_paths, args, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.args = args
        self.train = train

    """ Method used to apply transformation to images 
        and their corresponding masks."""
    def transform(self, image, mask):
         # transformation applied only if required and only to training images
        if self.args.apply_transformations == True and self.train == True:
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

        # input normalization if required
        if self.args.norm_input == True:
            image = TF.normalize(image, 
                                 mean=(0.5, 0.5, 0.5), 
                                 std=(0.5, 0.5, 0.5))

        return image, mask

    """ Method used to get (image, mask, label). """
    def __getitem__(self, index):
        # image_paths = tuple(path, label), target_paths = string(path)
        image_path = self.image_paths[index][0]
        label = self.image_paths[index][1]
        image = Image.open(image_path)
        mask = Image.open(self.target_paths[index])
        x, y = self.transform(image, mask)

        return x, y, label

    def __len__(self):
        return len(self.image_paths)
