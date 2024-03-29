###################################################################################################################################
# SOME REFERENCE: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/Basics/Imbalanced_classes #
##################################################################################################################################

from torch.utils.data import WeightedRandomSampler, DataLoader
from dataloader_utils import get_proportioned_dataset, EBHIDataset


""" Custom class that uses WeightedRandomSampler to sample from dataset 
    such that the model sees approximately each class the same number of times. """
class BalanceDataset(object):
    """ Initialize configurations. """
    def __init__(self, args, img_files_train, mask_files_train, train_lengths):
        self.args = args
        self.img_files_train = img_files_train
        self.mask_files_train = mask_files_train
        # this field contains the number of elements in each class
        self.train_lengths = train_lengths

    """ Method used to generates a well balanced trainloader: the pairs 
        (image, mask) selected for training are equal in number for the various classes. """
    def get_loader(self):
        print(f'\nPerforming data balancing...\n')
        
        # # Commentato perchè volendo applicare sia data augmentation che data balancing
        # # è necessario fornire il train-set aumentato. Inoltre, si evita di richiamare lo
        # # stesso metodo get_proportioned_dataset due volte. 
        # # Tuttavia, la riga di codice non viene rimossa poichè potrebbe essere utile per altri scopi
        # # futuri nel caso in cui BalanceDataset venga usata indipendentemente da get_proportioned_dataset prima.
        # img_files_train, mask_files_train, _, _, train_lengths, _ = get_proportioned_dataset(self.args)

        # forcing dataset creation to apply transformation if not applied with data-augmentation
        if self.args.dataset_aug == 0:
            self.args.apply_transformations = True
        train_dataset = EBHIDataset(self.img_files_train, self.mask_files_train, self.args, train=True)

        """ As the number of elements per class increases, the weight of 
            the class decreases, therefore the sampler considers more the 
            classes with a high weight value (small number of elements). """
        class_weights = []
        for lenght in self.train_lengths:
            class_weights.append(1/lenght)

        print(f'Class-weights: {class_weights}')

        # initialize a list with 'len(train_dataset)' zeros [0, ..., 0] 
        sample_weights = [0] * len(train_dataset)

        # for each element in the 'train_set', the corresponding class 
        # importance weight is associated in order. Therefore, when using the sampler, 
        # 'shuffle' is set to False.
        for idx, (_, _, label) in enumerate(train_dataset):
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight

        """ It is possible for a given sample to be included more than once (replacement=True).
            Weights is a sequence of weights, not necessary summing up to one. num_samples is
            the number of samples to draw. """
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        # generating the train-loader using the defined sampler
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.bs_train,
                                      num_workers=self.args.workers, sampler=sampler)

        return train_dataloader, class_weights


""" Example of use: 
import torch
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--bs_train', type=int, default=4,
                        help='number of elements in training batch')
    parser.add_argument('--workers', type=int, default=2,
                        help='number of workers in data loader')

    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')

    parser.add_argument('--random_seed', type=int, default=1,
                        help='random seed used to generate random train and test set')
    parser.add_argument('--norm_input', action='store_true',
                        help='normalize input images')

    parser.add_argument('--th', type=int, default=300,
                        help='threshold used to manipulate the dataset-%-split')
    parser.add_argument('--dataset_path', type=str, default='./data/EBHI-SEG/',
                        help='path were to save/get the dataset')

    parser.add_argument('--apply_transformations', action='store_true',
                        help='Apply some transformations to images and corresponding masks')

    return parser.parse_args()


def main(args):
    balance_d = BalanceDataset(args)
    train_dataloader, _ = balance_d.get_loader()

    num_normal = 0          # label - 0
    num_polyp = 0            # label - 1
    num_lowgradein = 0       # label - 2
    num_highgradein = 0      # label - 3
    num_adenocarcinoma = 0   # label - 4
    num_serratedadenoma = 0  # label - 5

    for epoch in range(args.epochs):

        loop = tqdm(enumerate(train_dataloader),
                    total=len(train_dataloader), leave=True)

        for batch, (_, _, labels) in loop:
            num_normal += torch.sum(labels == 0)
            num_polyp += torch.sum(labels == 1)
            num_lowgradein += torch.sum(labels == 2)
            num_highgradein += torch.sum(labels == 3)
            num_adenocarcinoma += torch.sum(labels == 4)
            num_serratedadenoma += torch.sum(labels == 5)
        
        print('')

    print(f'\nWeightedRandomSampler sample from dataset such that the model ' +
          f'sees approximately each class the same number of times: \n')

    print(f'Normal-class --> images sampled: {num_normal.item()}')
    print(f'Polyp-class images --> sampled: {num_polyp.item()}')
    print(f'Low-grade IN-class --> images sampled: {num_lowgradein.item()}')
    print(f'High-grade IN-class --> images sampled: {num_highgradein.item()}')
    print(f'Adenocarcinoma-class --> images sampled: {num_adenocarcinoma.item()}')
    print(f'Serrated adenoma-class --> images sampled: {num_serratedadenoma.item()}')


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
"""
