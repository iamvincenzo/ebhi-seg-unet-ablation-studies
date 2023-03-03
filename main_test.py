import os
import torch
import datetime
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from solver import Solver
from data_aug import AugmentData
from data_cleaning import clean_dataset, remove_aug
from dataloader_utils import balanced_dataset, EBHIDataset
from plotting_utils import set_default, plot_samples, add_sample_hist


def check_args_integrity(args, tn_l, te_l):
    """ This function is used to check the validity of some cmd parameters.
    Args:
        args (class 'argparse.Namespace'): cmd parameters
        tn_l (class 'int'): train-set length
        te_l (class 'int'): test-set length
    """

    if (args.bs_train < 0):
        print('\nError: bs_train must be positive!')
        os._exit(1)
    if (args.bs_train > tn_l):
        print('\nError: bs_train must be smaller than len(train-set)!')
        os._exit(1)
    if (args.bs_test < 0):
        print('\nError: bs_test must be positive!')
        os._exit(1)
    if (args.bs_test > te_l):
        print('\nError: bs_test must be smaller than len(test-set)!')
        os._exit(1)
    if (args.epochs < 0):
        print('\nError: epochs must be positive!')
        os._exit(1)
    if (args.print_every < 0):
        print('\nError: print_every must be positive!')
        os._exit(1)
    if (args.th < 0):
        print('\nError: th must be positive!')
        os._exit(1)
    if (args.pretrained_net == True and args.online_impl_net == True):
        print('\nError: only one between pretrained_net and online_impl_net!')
        os._exit(1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        default="run_0", help='name of current run')
    parser.add_argument('--model_name', type=str, default="first_train",
                        help='name of the model to be saved/loaded')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
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
                        help='list of features value', default=[64, 128, 256, 512])  # default=['64', '128', '256', '512'])
    parser.add_argument('--online_impl_net', action='store_true',
                        help='load online model implementation')

    parser.add_argument('--apply_transformations', action='store_true',
                        help='Apply some transformations to images ad masks')

    parser.add_argument('--dataset_aug', type=int, default=0,
                        help='Data augmentation of each class')

    return parser.parse_args()


def main(args):
    log_folder = './runs/' + args.run_name + '_' + \
        datetime.datetime.now().strftime('%d%m%Y-%H%M%S')
    writer = SummaryWriter(log_folder)

    set_default()

    """ Dataset cleaning:
        it has to be executed only the first time you unzip dataset 
        because Low-grade IN: 639(imgs), 637(masks). """
    # clean_dataset(args)

    """ Data augmentation with albumentations. """
    remove_aug(args) # remove augmented images if in the previous experiment we used data augmentation

    """ Creation of a well balanced dataset: 
        You need to take a certain percentage of examples (randomly) from each 
        class for the train and test/validation set, 80% and 20% respectively.

        Specifically, this function returns some lists of paths for image files, mask files
        both for train and test/validation set.
        
        - w_train_clss is a list that contains the number of elements for each class in train-set.
        - w_test_clss is a list that contains the number of elements for each class in test set.         
        
        N.B: different random seeds generate different train and test/validation sets."""
    img_files_train, mask_files_train, img_files_test, mask_files_test, w_train_clss, w_test_clss = balanced_dataset(args)
 
    """ Dataset augmentation, so so you need to create again training 
        image/mask set with augmented images (created images). """
    if args.dataset_aug > 0:
        AugmentData(img_files_train, mask_files_train, args)
        img_files_train, mask_files_train, img_files_test, mask_files_test, w_train_clss, w_test_clss = balanced_dataset(args)

    """ Adding histograms to tensorboard to represent the number of samples 
        for each class, both for train and test/validation set. """
    writer.add_figure('train_per_class_histo',
                      add_sample_hist(w_train_clss, 'train'))
    writer.add_figure('test_per_class_histo',
                      add_sample_hist(w_test_clss, 'test'))
    writer.close()

    # showing some samples
    print(f'First sample in train-set: {img_files_train[0]}')
    print(f'First sample in test-set: {img_files_test[0]}\n')

    """ Check the validity of some cmd parameters. """
    check_args_integrity(args, len(img_files_train), len(img_files_test))

    # Dataset stores all your data, and Dataloader is can be used to iterate through the data,
    # manage batches, transform the data, and much more.
    """ Custom dataset to load dataset that comes with additional information (mask):
        Dataset stores the samples and their corresponding labels. """
    train_dataset = EBHIDataset(img_files_train, mask_files_train, args)
    test_dataset = EBHIDataset(img_files_test, mask_files_test, args)

    """ DataLoader wraps an iterable around the Dataset to enable easy access to the samples
        according to a specific batch-size (load the data in memory). """
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.bs_train, shuffle=True, num_workers=args.workers)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.bs_test, shuffle=True, num_workers=args.workers)

    """ Specify the device type responsible to load a tensor into memory. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    """ Get some train-set samples (images) and display them
        in tensorboard. """
    images, masks, labels = next(iter(train_dataloader))
    # images = images.to(device) ???
    # masks = masks.to(device) ????
    writer.add_figure('ebhi-seg_samples',
                      plot_samples(images, masks, labels, args))
    writer.close()

    # define solver class
    solver = Solver(train_loader=train_dataloader,
                    test_loader=test_dataloader,
                    device=device,
                    writer=writer,
                    args=args)

    # TRAIN model
    solver.train()


if __name__ == "__main__":
    args = get_args()
    # If folder doesn't exist, then create it.
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    print(args)
    main(args)
