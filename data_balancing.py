##################################################################################################################################
# SOME REFERENCE: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/Basics/Imbalanced_classes #
##################################################################################################################################

from torch.utils.data import WeightedRandomSampler, DataLoader
from dataloader_utils import get_proportioned_dataset, EBHIDataset


""" Custom class that uses 'WeightedRandomSampler' to sample 
    from dataset such that the model sees approximately each
    class the same number of times. """
class BalanceDataset():
    def __init__(self, args):
        super(BalanceDataset, self).__init__()

        self.args = args

    """ This method is used to generates a well balanced 
        trainloader: the pairs (image, mask) selected 
        for training are equal in number for the 
        various classes. """
    def get_loader(self):
        img_files_train, mask_files_train, _, _, w_train_clss, _ = get_proportioned_dataset(self.args)

        self.args.apply_transforms = True # forcing dataset creation to apply transformation
        train_dataset = EBHIDataset(
            img_files_train, mask_files_train, self.args)

        """ As the number of elements per class increases, the weight of 
            the class decreases, therefore the sampler considers more the 
            classes with a high weight value (small number of elements). """
        class_weights = []
        for lenght in w_train_clss:
            class_weights.append(1 / lenght) 

        print(f'Class-weights: {class_weights}')

        sample_weights = [0] * len(train_dataset)

        for idx, (_, _, label) in enumerate(train_dataset):
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight

        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True)

        train_dataloader = DataLoader(train_dataset, batch_size=self.args.bs_train,
                                      num_workers=self.args.workers, sampler=sampler)

        return train_dataloader, class_weights



""" Example of use 
import torch
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--bs_train', type=int, default=4,
                        help='number of elements in training batch')
    parser.add_argument('--bs_test', type=int, default=1,
                        help='number of elements in test batch')
    parser.add_argument('--workers', type=int, default=2,
                        help='number of workers in data loader')
    parser.add_argument('--print_every', type=int, default=445,
                        help='print losses every N iteration')

    parser.add_argument('--random_seed', type=int, default=1,
                        help='random seed used to generate random train and test set')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--loss', type=str, default='dc_loss',
                        choices=['dc_loss', 'jac_loss'], help='loss function used for optimization')
    parser.add_argument('--opt', type=str, default='SGD',
                        choices=['SGD', 'Adam'], help='optimizer used for training')

    parser.add_argument('--early_stopping', type=int, default=5,
                        help='threshold used to manipulate the early stopping epoch tresh')

    parser.add_argument('--norm_input', action='store_true',
                        help='normalize input images')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='use batch normalization layers in model')
    parser.add_argument('--use_double_batch_norm', action='store_true',
                        help='use batch normalization layers in model')
    parser.add_argument('--use_inst_norm', action='store_true',
                        help='use instance normalization layers in model')
    parser.add_argument('--use_double_inst_norm', action='store_true',
                        help='use instance normalization layers in model')

    parser.add_argument('--th', type=int, default=300,
                        help='threshold used to manipulate the dataset-%-split')
    parser.add_argument('--dataset_path', type=str, default='./data/EBHI-SEG/',
                        help='path were to save/get the dataset')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./model_save', help='path were to save the trained model')

    parser.add_argument('--resume_train', action='store_true',
                        help='load the model from checkpoint before training')

    parser.add_argument('--pretrained_net', action='store_true',
                        help='load pretrained model on BRAIN MRI')

    parser.add_argument('--features', nargs='+',
                        help='list of features value', default=[64, 128, 256, 512])
    parser.add_argument('--arc_change_net', action='store_true',
                        help='load online model implementation')

    parser.add_argument('--apply_transformations', action='store_true',
                        help='Apply some transformations to images ad masks')

    parser.add_argument('--dataset_aug', type=int, default=0,
                        help='Data augmentation of each class')

    return parser.parse_args()


def main(args):
    balance_d = BalanceDataset(args)
    train_dataloader = balance_d.get_loader()

    num_normal = 0          # label - 0
    num_polyp = 0            # label - 1
    num_lowgradein = 0       # label - 2
    num_highgradein = 0      # label - 3
    num_adenocarcinoma = 0   # label - 4
    num_serratedadenoma = 0  # label - 5

    loop = tqdm(enumerate(train_dataloader),
                total=len(train_dataloader), leave=False)

    for epoch in range(10):
        for batch, (_, _, labels) in loop:
            num_normal += torch.sum(labels == 0)
            num_polyp += torch.sum(labels == 1)
            num_lowgradein += torch.sum(labels == 2)
            num_highgradein += torch.sum(labels == 3)
            num_adenocarcinoma += torch.sum(labels == 4)
            num_serratedadenoma += torch.sum(labels == 5)

    print(f'\nWeightedRandomSampler sample from dataset such that the model ' +
          f'sees approximately each class the same number of times: ')

    print(f'Num Normal: {num_normal.item()}')
    print(f'Num Polyp: {num_polyp.item()}')
    print(f'Num Low-grade IN: {num_lowgradein.item()}')
    print(f'Num High-grade IN: {num_highgradein.item()}')
    print(f'Num Adenocarcinoma: {num_adenocarcinoma.item()}')
    print(f'Num Serrated adenoma: {num_serratedadenoma.item()}')


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
"""