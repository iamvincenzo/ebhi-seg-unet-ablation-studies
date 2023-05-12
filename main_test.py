import os
import torch
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from solver import Solver
from data_cleaning import remove_aug
from data_augmentation import AugmentData
from data_balancing import BalanceDataset
from dataloader_utils import EBHIDataset
from dataloader_utils import get_proportioned_dataset, classes
from plotting_utils import set_default, plot_samples, add_sample_hist, add_metric_hist


""" Helper function used to check 
    the validity of some cmd parameters. """
def check_args_integrity(args, tn_l, te_l):
    if (args.bs_train < 0):
        print('\nError: bs_train must be positive!')
        os._exit(1)
    if (args.bs_train > tn_l):
        print('\nError: bs_train must be smaller than train-set lenght!')
        os._exit(1)
    if (args.bs_test < 0):
        print('\nError: bs_test must be positive!')
        os._exit(1)
    if (args.bs_test > te_l):
        print('\nError: bs_test must be smaller than test-set lenght!')
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
    if (args.pretrained_net == True and args.arc_change_net == True):
        print('\nError: only one between pretrained_net and arc_change_net!')
        os._exit(1)
    if (args.dataset_aug < 0):
        print('\nError: dataset_aug must be positive!')
        os._exit(1)
    if (args.random_structured == True and args.random_unstructured == True):
        print('\nError: choose only one between random_structured and random_unstructured!')
        os._exit(1)
    if (args.single_mod == True and args.double_mod == True):
        print('\nError: choose only one between single_mod and double_mod!')
        os._exit(1)
    if ((args.selective_ablation == True and args.random_structured == False) and 
        (args.selective_ablation == True and args.random_unstructured == False)):        
            print(f'\nError: at least one of random_structured or random_unstructured has to ' 
                  f'be selected with "selective_ablation"!')
            os._exit(1)
    if ((args.all_one_by_one == True and args.random_structured == False) and 
        (args.all_one_by_one == True and args.random_unstructured == False)):        
            print(f'\nError: at least one of random_structured or random_unstructured has to '
                  f'be selected with "all_one_by_one"!')
            os._exit(1)
    if ((args.all_one_by_one == True and args.single_mod == True) or 
        (args.all_one_by_one == True and args.double_mod == True)):        
            print('\nError: you can\'t select all_one_by_one with single_mod/double_mod!')
            os._exit(1)
    
""" Helper function used to get cmd parameters. """
def get_args():
    parser = argparse.ArgumentParser()

    # model-infos
    ###################################################################
    parser.add_argument('--run_name', type=str, default="run_0", 
                        help='the name assigned to the current run')
    parser.add_argument('--model_name', type=str, default="first_train",
                        help='the name of the model to be saved or loaded')
    ###################################################################

    # training-parameters (1)
    ###################################################################
    parser.add_argument('--epochs', type=int, default=100,
                        help='the total number of training epochs')
    parser.add_argument('--bs_train', type=int, default=4,
                        help='the batch size for training data')
    parser.add_argument('--bs_test', type=int, default=1,
                        help='the batch size for test data')
    parser.add_argument('--workers', type=int, default=2,
                        help='the number of workers in the data loader')
    parser.add_argument('--print_every', type=int, default=445,
                        help='the frequency of printing losses during training')
    ###################################################################

    # training-parameters (2)
    ###################################################################
    parser.add_argument('--random_seed', type=int, default=1,
                        help='the random seed used to ensure reproducibility')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='the learning rate for optimization')
    parser.add_argument('--loss', type=str, default='dc_loss',
                        choices=['dc_loss', 'jac_loss', 'bcewl_loss', 'custom_loss'],
                        help='the loss function used for model optimization')
    parser.add_argument('--val_custom_loss', type=float, default=0.3,
                        help='the weight of the jac_loss in the overall custom-loss function')
    parser.add_argument('--opt', type=str, default='SGD', choices=['SGD', 'Adam'], 
                        help='the optimizer used for training')
    parser.add_argument('--early_stopping', type=int, default=5,
                        help='the threshold for early stopping during training')
    ###################################################################

    # training-parameters (3)
    ###################################################################
    parser.add_argument('--resume_train', action='store_true',
                        help='determines whether to load the model from a checkpoint')
    ###################################################################

    # network-architecture-parameters (1) - normalization: 
    # you can modify network architecture
    ###################################################################
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='indicates whether to use batch normalization layers in each conv layer')
    parser.add_argument('--use_double_batch_norm', action='store_true',
                        help='indicates whether to use 2 batch normalization layers in each conv layer')
    parser.add_argument('--use_inst_norm', action='store_true',
                        help='indicates whether to use instance normalization layers in each conv layer')
    parser.add_argument('--use_double_inst_norm', action='store_true',
                        help='indicates whether to use 2 instance normalization layers in each conv layer')
    parser.add_argument('--weights_init', action='store_true',
                        help='determines whether to use weights initialization')
    ###################################################################

    # data-parameters
    ###################################################################
    parser.add_argument('--th', type=int, default=300,
                        help='the threshold used to split the dataset into train and test subsets')
    parser.add_argument('--dataset_path', type=str, default='./data/EBHI-SEG/',
                        help='the path to save or retrieve the dataset')
    parser.add_argument('--checkpoint_path', type=str, default='./model_save', 
                        help='the path to save the trained model')
    ###################################################################

    # model-types
    ###################################################################
    parser.add_argument('--pretrained_net', action='store_true',
                        help='indicates whether to load a pretrained model on BRAIN-MRI')
    # this model architecture can be modified.
    # arc_change_net can be used with standard dice_loss/jac_loss or bce_with_logits_loss
    parser.add_argument('--arc_change_net', action='store_true',
                        help='indicates whether to load a customizable model')
    # num of filters in each convolutional layer (num of channels of the feature-map): 
    # so you can modify the network architecture
    parser.add_argument('--features', nargs='+',  default=[64, 128, 256, 512],
                        help='a list of feature values (number of filters i.e., neurons in each layer)')
    ###################################################################

    # data transformation
    ###################################################################
    parser.add_argument('--norm_input', action='store_true',
                        help='indicates whether to normalize the input data')
    ###################################################################

    # data-manipulation 
    # (hint: select only one of the options below)
    ###################################################################
    parser.add_argument('--apply_transformations', action='store_true',
                        help='indicates whether to apply transformations to images and corresponding masks')
    parser.add_argument('--dataset_aug', type=int, default=0,
                        help='determines the type of data augmentation applied to each class')
    parser.add_argument('--balanced_trainset', action='store_true',
                        help='generates a well-balanced train_loader for training')
    ###################################################################

    # ablation-Studies (1) - types
    ###################################################################
    parser.add_argument('--global_ablation', action='store_true',
                        help='initiates an ablation study using global_unstructured or l1_unstructured')
    parser.add_argument('--grouped_pruning', action='store_true',
                        help='initiates an ablation study using global_unstructured')
    ###################################################################
    parser.add_argument('--all_one_by_one', action='store_true',
                        help='initiates an ablation study using random_structured or random_unstructured')
    ###################################################################
    parser.add_argument('--selective_ablation', action='store_true',
                        help='initiates an ablation study using random_structured or random_unstructured')
    parser.add_argument('--single_mod', action='store_true',
                        help='initiates an ablation study with only one module')
    parser.add_argument('--double_mod', action='store_true',
                        help='initiates an ablation study with two modules')
    parser.add_argument('--random_structured', action='store_true',
                        help='initiates an ablation study using random_structured')
    parser.add_argument('--random_unstructured', action='store_true',
                        help='initiates an ablation study using random_unstructured')
    ###################################################################

    # ablation-Studies (2) - parameters
    ###################################################################
    parser.add_argument('--conv2d_prune_amount', type=float, default=0.25,
                        help='the amount of pruning applied to conv2d layers')
    parser.add_argument('--linear_prune_amount', type=float, default=0.2,
                        help='the amount of pruning applied to linear layers')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='the number of iterations for the pruning process')
    ###################################################################

     # weights-analysis 
    ###################################################################
    parser.add_argument('--weights_distr_histo', action='store_true',
                        help='plots the histogram of weights distribution')
    parser.add_argument('--plt_weights_distr', action='store_true',
                        help='plots the filters as iamges to visualize CNN kernels')
    ###################################################################

    return parser.parse_args()

""" Main function used to run the experiment/analysis. """
def main(args):
    # tensorboard specifications
    date =  '_' + datetime.now().strftime('%d%m%Y-%H%M%S')
    writer = SummaryWriter('./runs/' + args.run_name + date)

    # setting fig-style
    set_default() 

    # """ Dataset cleaning:
    #     it has to be executed only the first time you unzip dataset 
    #     because Low-grade IN: 639(imags), 637(masks). """
    # from data_cleaning import clean_dataset
    # clean_dataset(args) 

    """ Data augmentation with albumentations: remove augmented 
        images if in the previous experiment we used data augmentation. """
    remove_aug(args)

    ##################################
    # N.B: balanced != proportionate #
    ##################################
    """ Creation of a well-proportionate dataset:
        You need to take a certain percentage of examples (randomly) from each 
        class for the train and test(validation) set, 80% and 20% respectively.

        Specifically, this function returns some lists of paths for (image files, mask files)
        both for train and test(validation) set.
            - train_lengths is a list that contains the number of elements for each class in train-set.
            - test_lengths is a list that contains the number of elements for each class in test set.         
        
        N.B: different random seeds generate different train and test/validation sets. 
    """
    (img_files_train, mask_files_train, img_files_test, mask_files_test, 
        train_lengths, test_lengths) = get_proportioned_dataset(args.dataset_path, 
                                                                classes, args.th, 
                                                                args.random_seed)
    
    """ Dataset augmentation, so you need to create again training 
        image/mask set with augmented images:
            AugmentData generates new images/masks and save them in the
            same directory of the original files. """
    if args.dataset_aug > 0:
        aug = AugmentData(args, img_files_train, mask_files_train)
        aug_img_files_train, aug_mask_files_train = aug.get_augmented_train_set()
        print(f'Number of elements in train-set before data-augmentation: {len(img_files_train)}')
        print(f'Number of augmented images: {len(aug_img_files_train)}\n')
        img_files_train += aug_img_files_train
        mask_files_train += aug_mask_files_train
        print(f'Number of elements in train-set after data-augmentation: '
              f'img={len(img_files_train)}, msk={len(mask_files_train)}\n')
        
        # """ The most common practice is to apply data augmentation only to the training samples. The reason 
        #     is that we want to increase our model's generalization performance by adding more data and diversifying 
        #     the training dataset. However, we can also use it during testing. """
        # aug = AugmentData(args, img_files_train+img_files_test, mask_files_train+mask_files_test)
        # # so you need to regenerate the entire dataset: train and test(validation)
        # (img_files_train, mask_files_train, img_files_test, mask_files_test, 
        # train_lengths, test_lengths) = get_proportioned_dataset(args.dataset_path, 
        #                                                         classes, args.th, 
        #                                                         args.random_seed)

    """ Generates a well balanced train_loader used to train the network. """
    if args.balanced_trainset == True:
        balance_d = BalanceDataset(args, img_files_train, mask_files_train, train_lengths)
        train_dataloader, class_weights = balance_d.get_loader()
        print(f'\nBalanced train-dataloader created...\n')

        # adding histograms to tensorboard to represent the 
        # weight of each class in creating the balanced train_loader
        writer.add_figure('weights_balanced_trainloader', add_metric_hist(class_weights, 'weights'))

    # adding histograms to tensorboard to represent the number of 
    # samples for each class, both for train and test(validation) set
    writer.add_figure('train_per_class_histo', add_sample_hist(train_lengths, 'train'))
    writer.add_figure('test_per_class_histo', add_sample_hist(test_lengths, 'test'))
    writer.close()

    # showing some samples
    print(f'First sample in train-set: {img_files_train[0]}, {mask_files_train[0]}')
    print(f'First sample in test-set: {img_files_test[0]}, {mask_files_test[0]}\n')

    # Check the validity of some cmd parameters
    check_args_integrity(args, len(img_files_train), len(img_files_test))

    # Dataset stores all the data, and Dataloader is can be used to 
    # iterate through the data, manage batches, transform the data, and much more.
    """ Custom dataset to load dataset that comes with additional 
        information (mask): Dataset stores the samples and their corresponding labels. """
    if args.balanced_trainset == False:
        # the train_dataset is already generated inside 'BalanceDataset'
        train_dataset = EBHIDataset(img_files_train, mask_files_train, args, train=True)
    test_dataset = EBHIDataset(img_files_test, mask_files_test, args, train=False)

    # DataLoader wraps an iterable around the Dataset to enable easy 
    # access to the samples according to a specific batch-size (load the data in memory)
    if args.balanced_trainset == False:
        # the train_dataloader is already generated inside 'BalanceDataset'
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs_train, shuffle=True, num_workers=args.workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs_test, shuffle=True, num_workers=args.workers)

    # specify the device type responsible to load a tensor into memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # get some train-set samples (images) and display them in tensorboard
    images, masks, labels = next(iter(train_dataloader))
    writer.add_figure('ebhi-seg_samples', plot_samples(images, masks, labels, args))
    writer.close() 

    # solver class used to train the model and execute ablation studies
    solver = Solver(train_loader=train_dataloader,
                    test_loader=test_dataloader,
                    device=device,
                    writer=writer,
                    args=args)
        
    # diffetent types of ablation-studies
    if (args.global_ablation == True or args.selective_ablation == True 
        or args.all_one_by_one == True):
        solver.start_ablation_study()
    # debug: plot the weights distribution
    elif args.weights_distr_histo == True:
        solver.weights_distribution_analysis()
    else:
        solver.train()    
       
""" Starting the simulation. """ 
if __name__ == "__main__":
    args = get_args()

    # if folder doesn't exist, then create it
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.isdir('statistics'):
        os.makedirs('statistics')
    if not os.path.isdir('abl_statistics'):
        os.makedirs('abl_statistics')

    print(f'\n{args}')
    main(args)
