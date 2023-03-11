import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from model import UNet
from main_test import get_args
import matplotlib.pyplot as plt
from arc_change_net import UNET
import torchvision.transforms as T
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from plotting_utils import plot_check_results
from dataloader_utils import get_proportioned_dataset, EBHIDataset
from metrics import dc_loss, jac_loss, custom_loss, binary_jac, binary_acc, binary_prec, binary_rec, binary_f1s


class AnalyzeNetwork(object):
    def __init__(self, args):
        super(AnalyzeNetwork, self).__init__()

        self.args = args
        self.model_name = 'ebhi-seg_u-net_{}.pth'.format(self.args.model_name)

        self.removed_info = []

        if self.args.pretrained_net == True:
            model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                   in_channels=3, out_channels=1, init_features=32, pretrained=True)
            print(f'\nPretrained model implementation selected:\n\n {model}')
        elif self.args.arc_change_net == True:
            model = UNET(self.args, 3, 1, [int(f) for f in self.args.features])
            print(f'\nOnline model implementation selected:\n\n {model}')
        else:
            model = UNet(self.args)
            print(f'\nStandard model implementation selected:\n\n {model}')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Device: {device}')

        self.device = device

        self.net = model.to(self.device)

        # define Loss function
        if self.args.loss == 'dc_loss':
            self.criterion = dc_loss
            print(f'\nDC_loss selected!\n')
        elif self.args.loss == 'jac_loss':
            self.criterion = jac_loss
            print(f'\nJAC_loss selected!\n')
        elif self.args.loss == 'bcewl_loss' and self.args.arc_change_net == True:
            self.criterion = nn.BCEWithLogitsLoss()
            print(f'\nBCEWithLogitsLoss selected!\n')
        elif self.args.loss == 'custom_loss':
            self.criterion = custom_loss
            print(f'\custom_loss selected!\n')

        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(
            check_path, map_location=torch.device(self.device)))
        print('\nModel loaded!\n')

        ###########################################################################
        self.test_loader = self.create_val_set()
        self.state_dict_mod = self.net.state_dict()
        ###########################################################################

    def add_gradient_hist_mod(self, net):
        ave_grads = []
        layers = []
        for n, p in net.named_parameters():
            if ('bias' not in n):
                layers.append(n)
                if p.requires_grad:  # indicates whether a variable is trainable
                    ave_grad = np.abs(p.detach().numpy()).mean()
                else:
                    ave_grad = 0
                ave_grads.append(ave_grad)

        layers = [layers[i].replace(".weight", "") for i in range(len(layers))]

        fig = plt.figure(figsize=(12, 12))
        plt.bar(np.arange(len(ave_grads)), ave_grads, lw=1, color="b")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90)
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.legend([Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['mean-gradient', 'zero-gradient'])
        plt.tight_layout()

        return fig

    def binarization_tensor(self, mask, pred):
        transform = T.ToPILImage()
        binary_mask = transform(np.squeeze(mask)).convert('1')
        binary_pred = transform(np.squeeze(pred)).convert('1')
        maskf = TF.to_tensor(binary_mask).view(-1)
        predf = TF.to_tensor(binary_pred).view(-1)

        return maskf, predf

    def test(self):
        print(f'\nPerforming validation-test after filters removal...')

        if self.args.pretrained_net == True:
            model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                   in_channels=3, out_channels=1, init_features=32, pretrained=True)
            print(f'\nPretrained model implementation selected:\n\n {model}')

        elif self.args.arc_change_net == True:
            model = UNET(self.args, 3, 1, [int(f) for f in self.args.features])
            print(f'\nOnline model implementation selected:\n\n {model}')

        else:
            model = UNet(self.args)
            print(f'\nStandard model implementation selected:\n\n {model}\n\n')

        print(f'\nModel name: {self.model_name}')
        a = input('Premi un tasto per continuare...\n')

        # loading the model with zeros
        self.net.load_state_dict(self.state_dict_mod)

        test_losses = []

        self.net.eval()  # put net into evaluation mode

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            # compute some metrics performance for each class on the test set
            if self.args.bs_test == 1:
                dc_class_test = list([[], [], [], [], [], []])
                jac_cust_class_test = list([[], [], [], [], [], []])
                jac_class_test = list([[], [], [], [], [], []])
                acc_class_test = list([[], [], [], [], [], []])
                prec_class_test = list([[], [], [], [], [], []])
                rec_class_test = list([[], [], [], [], [], []])
                f1s_class_test = list([[], [], [], [], [], []])

            test_loop = tqdm(enumerate(self.test_loader),
                             total=len(self.test_loader), leave=False)

            for batch_test, (test_images, test_targets, test_labels) in test_loop:
                test_images = test_images.to(self.device)
                test_targets = test_targets.to(self.device)
                test_pred = self.net(test_images.detach())

                # general loss
                test_loss = self.criterion(test_pred, test_targets).item()
                test_losses.append(test_loss)

                # used to check model improvement
                self.check_results(batch_test)

                if self.args.bs_test == 1:
                    # function that binarize an image and then convert it to a tensor
                    maskf, predf = self.binarization_tensor(
                        test_targets, test_pred)

                    # per class metrics
                    dc_class_test[test_labels].append(1 - test_loss)
                    jac_cust_class_test[test_labels].append(
                        1 - jac_loss(test_pred, test_targets).item())
                    jac_class_test[test_labels].append(
                        binary_jac(maskf, predf))
                    acc_class_test[test_labels].append(
                        binary_acc(maskf, predf))
                    prec_class_test[test_labels].append(
                        binary_prec(maskf, predf))
                    rec_class_test[test_labels].append(
                        binary_rec(maskf, predf))
                    f1s_class_test[test_labels].append(
                        binary_f1s(maskf, predf))

            batch_avg_test_loss = np.average(test_losses)

            print(f'\nvalid_loss: {batch_avg_test_loss:.5f}\n')

            if self.args.bs_test == 1:
                print(
                    f'Dice for class {[np.average(x) for x in dc_class_test]}')
                print(
                    f'Jaccard index custom for class {[np.average(x) for x in jac_cust_class_test]}')
                print(
                    f'Binary Jaccard index for class {[np.average(x) for x in jac_class_test]}')
                print(
                    f'Binary Accuracy for class {[np.average(x) for x in acc_class_test]}')
                print(
                    f'Binary Precision for class {[np.average(x) for x in prec_class_test]}')
                print(
                    f'Binary Recall for class {[np.average(x) for x in rec_class_test]}')
                print(
                    f'Binary F1-score for class {[np.average(x) for x in f1s_class_test]}')

            check_path = os.path.join(
                self.args.checkpoint_path, 'abl_' + self.model_name)
            torch.save(self.net.state_dict(), check_path)
            print('\nModel saved!\n')

            # print(f'Different conv-filters removed from each conv-layer: \n\n')
            # for x in self.removed_info:
            #     print(x)

        # self.net.train()  # put again the model in trainining-mode

    def check_results(self, batch):
        with torch.no_grad():
            if batch % 50 == 49:
                self.net.eval()

                (img, mask, label) = next(iter(self.test_loader))
                img = img.to(self.device)
                mask = mask.to(self.device)
                mask = mask[0]
                pred = self.net(img)

                fig = plot_check_results(
                    img[0], mask, pred[0], label[0], self.args)

                plt.show(block=False)
                plt.pause(10)
                plt.close()

    ###########################################################################
    def create_val_set(self):
        # Il dataset di test è sempre lo stesso perchè c'è il parametro random-seed che consente
        # di eseguire gli stessi esperimenti.
        _, _, img_files_test, mask_files_test, _, _ = get_proportioned_dataset(
            self.args)
        test_dataset = EBHIDataset(img_files_test, mask_files_test, self.args)
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.args.bs_test, shuffle=True, num_workers=self.args.workers)

        return test_dataloader
    ###########################################################################

    ###########################################################################
    def filters_removal(self):  # [batch_size, channels, height, width]

        percentage = self.args.filters_removal_perc  # 0.15
        print(
            f'\nPerforming filters removal with percentage {percentage}...\n')

        random.seed(self.args.random_seed)  # for test repeatability

        for key, value in dict(self.net.named_parameters()).items():
            if ('weight' in key and value.dim() > 1):

                bs = value.size(dim=0)
                tot = int(percentage * bs)

                print(
                    f'Key: {key}, shape: {value.shape}, dim: {bs} - tot-conv_to_0: {tot}\n')

                extracted = []

                _val_ = value.clone()

                for random_idx in random.sample(range(0, bs), tot):
                    ####################################################################################################################                    #
                    # https://discuss.pytorch.org/t/how-remove-filter-from-layers-and-initialize-custom-filter-weights-to-layers/98870 #
                    ####################################################################################################################

                    if random_idx == 0:
                        _val_[:1] = 0
                        # print('if')
                    elif random_idx == bs:
                        _val_[bs:] = 0
                        # print('elif')
                    else:
                        _val_[random_idx:random_idx+1] = 0
                        # print('else')

                    # print(value.shape)
                    # a = input('')

                self.removed_info.append(extracted)

                self.state_dict_mod[key] = _val_  # model modification

                # print(f'Key: {key} - after_shape: {value.shape} - dim: {bs} - tot: {tot}\n\n')

                #######################################################################################
                # https://discuss.pytorch.org/t/how-to-drop-specific-filters-in-some-cnn-layers/40542 #
                #######################################################################################

        # self.check_grad_zero()
    ###########################################################################

    ###########################################################################
    def check_grad_zero(self):
        self.model_name = 'abl_' + self.model_name

        if self.args.pretrained_net == True:
            model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                   in_channels=3, out_channels=1, init_features=32, pretrained=True)
            print(f'\nPretrained model implementation selected:\n\n {model}')

        elif self.args.arc_change_net == True:
            model = UNET(self.args, 3, 1, [int(f) for f in self.args.features])
            print(f'\nOnline model implementation selected:\n\n {model}')

        else:
            model = UNet(self.args)
            print(f'\nStandard model implementation selected:\n\n {model}')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Device: {device}')

        self.net = model.to(device)

        # define Loss function
        if self.args.loss == 'dc_loss':
            self.criterion = dc_loss

        elif self.args.loss == 'jac_loss':
            self.criterion = jac_loss

        elif self.args.loss == 'bcewl_loss' and args.arc_change_net == True:
            self.criterion = nn.BCEWithLogitsLoss()

        check_path = os.path.join(args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(
            check_path, map_location=torch.device(device)))
        print('\nModel loaded!\n')

        percentage = self.args.filters_removal_perc  # 0.15

        print(f'\Checking filters removal with percentage {percentage}...\n')

        random.seed(self.args.random_seed)  # for test repeatability

        for key, value in dict(self.net.named_parameters()).items():
            if ('weight' in key and value.dim() > 1):
                print(key, value)
                # a = input('')
    ###########################################################################

    ###########################################################################
    def check_abl(self):
        self.model_name = 'abl_ebhi-seg_u-net_unet_16_0t.pth'

        model = UNET(args, 3, 1, [int(f) for f in self.args.features])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Device: {device}')

        self.net = model.to(device)

        # define Loss function
        if self.args.loss == 'dc_loss':
            self.criterion = dc_loss

        elif self.args.loss == 'jac_loss':
            self.criterion = jac_loss

        elif self.args.loss == 'bcewl_loss' and args.arc_change_net == True:
            self.criterion = nn.BCEWithLogitsLoss()

        check_path = os.path.join(args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(
            check_path, map_location=torch.device(device)))
        print(f'\nModel {self.model_name} loaded!\n')

        random.seed(args.random_seed)  # for test repeatability

        return self.net

        # for key, value in dict(net.named_parameters()).items():
        #     if ('weight' in key and value.dim() > 1):
        #         print(key, value)
        #         a = input('')
    ###########################################################################

    ###########################################################################    
    def check_normal(self):
        self.model_name = 'ebhi-seg_u-net_unet_16_0t.pth'

        model = UNET(args, 3, 1, [int(f) for f in self.args.features])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Device: {device}')

        self.net = model.to(device)

        # define Loss function
        if self.args.loss == 'dc_loss':
            self.criterion = dc_loss

        elif self.args.loss == 'jac_loss':
            self.criterion = jac_loss

        elif self.args.loss == 'bcewl_loss' and self.args.arc_change_net == True:
            self.criterion = nn.BCEWithLogitsLoss()

        check_path = os.path.join(args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(
            check_path, map_location=torch.device(device)))
        print(f'\nModel {self.model_name} loaded!\n')

        random.seed(self.args.random_seed)  # for test repeatability

        return self.net
    ###########################################################################

def get_args(argString):
    parser = argparse.ArgumentParser()

    # Model-infos
    ###################################################################
    parser.add_argument('--run_name', type=str,
                        default="run_0", help='name of current run')
    parser.add_argument('--model_name', type=str, default="first_train",
                        help='name of the model to be saved/loaded')
    ###################################################################

    # Training-parameters (1)
    ###################################################################
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
    ###################################################################

    # Training-parameters (2)
    ###################################################################
    parser.add_argument('--random_seed', type=int, default=1,
                        help='random seed used to generate random train and test set')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--loss', type=str, default='dc_loss',
                        choices=['dc_loss', 'jac_loss', 'bcewl_loss'],
                        help='loss function used for optimization')
    parser.add_argument('--opt', type=str, default='SGD',
                        choices=['SGD', 'Adam'], help='optimizer used for training')
    parser.add_argument('--early_stopping', type=int, default=5,
                        help='threshold used to manipulate the early stopping epoch tresh')
    ###################################################################

    # Training-parameters (3)
    ###################################################################
    parser.add_argument('--resume_train', action='store_true',
                        help='load the model from checkpoint before training')
    ###################################################################

    # Network-architecture-parameters (1) - normalization
    ###################################################################
    # data transformation
    parser.add_argument('--norm_input', action='store_true',
                        help='normalize input images')
    # you can modify network architecture
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='use batch normalization layers in each conv layer of the model')
    parser.add_argument('--use_double_batch_norm', action='store_true',
                        help='use 2 batch normalization layers in each conv layer of the model')
    parser.add_argument('--use_inst_norm', action='store_true',
                        help='use instance normalization layers in each conv layer of the model')
    parser.add_argument('--use_double_inst_norm', action='store_true',
                        help='use 2 instance normalization layers in each conv layer of the model')
    ###################################################################

    # Data-parameters
    ###################################################################
    parser.add_argument('--th', type=int, default=300,
                        help='threshold used to manipulate the dataset-%-split')
    parser.add_argument('--dataset_path', type=str, default='./data/EBHI-SEG/',
                        help='path were to save/get the dataset')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./model_save', help='path were to save the trained model')
    ###################################################################

    # Model-types
    ###################################################################
    parser.add_argument('--pretrained_net', action='store_true',
                        help='load pretrained model on BRAIN MRI')
    # This model architecture can be modified.
    # arc_change_net can be used with standard dice_loss/jac_loss or bce_with_logits_loss
    parser.add_argument('--arc_change_net', action='store_true',
                        help='load online model implementation')
    # num of filters in each convolutional layer (num of channels of the feature-map): so you can modify the network architecture
    parser.add_argument('--features', nargs='+',
                        help='list of features value', default=[64, 128, 256, 512])
    ###################################################################

    # Data-manipulation
    # (hint: select only one of the options below)
    ###################################################################
    parser.add_argument('--apply_transformations', action='store_true',
                        help='Apply some transformations to images and corresponding masks')
    parser.add_argument('--dataset_aug', type=int, default=0,
                        help='Data augmentation of each class')
    parser.add_argument('--balanced_trainset', action='store_true',
                        help='generates a well balanced train_loader')
    ###################################################################

    # TO DO ????
    ###################################################################
    parser.add_argument('--filters_removal_perc', type=float, default=0.3,
                        help='Data augmentation of each class')
    ###################################################################

    return parser.parse_args(argString.split())


def main(args):
    analyzeNn = AnalyzeNetwork(args)
    analyzeNn.filters_removal()
    analyzeNn.test()

    net = analyzeNn.check_abl()
    fig = analyzeNn.add_gradient_hist_mod(net)
    plt.show()

    net_norm = analyzeNn.check_normal()
    fig = analyzeNn.add_gradient_hist_mod(net_norm)
    plt.show()


if __name__ == "__main__":

    # ablation study over the best model
    argString = "--random_seed 0 --opt Adam --arc_change_net --use_double_inst_norm \
                 --balanced_trainset --early_stopping 3 --model_name unet_16_0t \
                 --filters_removal_perc 0.7"

    args = get_args(argString)
    main(args)
