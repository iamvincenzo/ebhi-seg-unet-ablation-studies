import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from model import UNet
from arc_change_net import UNET
from ablation_studies import AblationStudies
from metrics import dc_loss, jac_loss, custom_loss, binary_jac
from metrics import binary_acc, binary_prec, binary_rec, binary_f1s
from plotting_utils import plot_weights_distribution_histo
from plotting_utils import set_default, add_gradient_hist, add_metric_hist 
from plotting_utils import plot_check_results, kernels_viewer, activations_viewer


""" Solver for training and testing. """
class Solver(object):
    """ Initialize configurations. """
    def __init__(self, train_loader, test_loader, device, writer, args):
        self.args = args
        self.model_name = f'ebhi_seg_{self.args.model_name}.pth'

        # model selection
        if self.args.pretrained_net == True:
            model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                   in_channels=3, out_channels=1, init_features=32, pretrained=True)
            print(f'\nPretrained model implementation selected:\n\n {model}')
        elif self.args.arc_change_net == True:
            model = UNET(self.args, 3, 1, [int(f) for f in self.args.features])
            print(f'\nCustomizzable model implementation selected:\n\n {model}')
        else:
            model = UNet(self.args)
            print(f'\nStandard model implementation selected:\n\n {model}')

        # model definition
        self.net = model.to(device)

        # load a pretrained model for 
        # resuming training or to execute ablation studies
        if (self.args.resume_train == True or 
            self.args.global_ablation == True or 
            self.args.selective_ablation == True or 
            self.args.all_one_by_one == True):
            self.load_model(device)

        # loss function definition
        if self.args.loss == 'dc_loss':
            self.criterion = dc_loss
            print(f'\nDC_loss selected...')
        elif self.args.loss == 'jac_loss':
            self.criterion = jac_loss
            print(f'\nJAC_loss selected...')
        elif (self.args.loss == 'bcewl_loss' 
              and self.args.arc_change_net == True):
            self.criterion = nn.BCEWithLogitsLoss()
            print(f'\nBCEWithLogitsLoss selected...')
        elif self.args.loss == 'custom_loss':
            self.criterion = custom_loss
            print(f'\ncustom_loss selected...')

        # optimizer definition
        if self.args.opt == "SGD":
            self.optimizer = optim.SGD(self.net.parameters(), 
                                       lr=self.args.lr, momentum=0.9)
        elif self.args.opt == "Adam":
            self.optimizer = optim.Adam(self.net.parameters(), 
                                        lr=self.args.lr, betas=(0.9, 0.999))

        # other training/validation parameters
        self.epochs = self.args.epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.writer = writer

        # visualize the model built on TensorBoard
        images, _, _ = next(iter(self.train_loader))
        self.writer.add_graph(self.net, images.to(self.device))
        self.writer.close()

        # setting fig-style
        set_default() 

    """ Method used to save the model. """
    def save_model(self):
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        torch.save(self.net.state_dict(), check_path)
        print('\nModel saved...\n')

    """ Method used to load the model. """
    def load_model(self, device):
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(check_path, 
                                            map_location=torch.device(device)))
        print('\nModel loaded...\n')

    """ Method used to save collected training-statistics. """
    def save_json(self, file):
        with open('./statistics/my_dic_train_results_' + self.args.model_name + 
                  '_' + f'{datetime.now():%d%m%Y-%H%M%S}' + '.json', 'w') as f:
            json.dump(file, f)

    """ Method used to binarize a tensor (mask)
        in order to compute binary accuracy, precision, recall, f1-score. 
        The convert('1') function creates a thresholded image where all pixels with values 
        less than 128 are set to 0, and all pixels with values greater than or equal to 128 
        are set to 1. """
    def binarization_tensor(self, mask, pred):
        transform = T.ToPILImage()
        # np.squeeze(mask) 
        # --> [1, 3, 224, 224] --> [3, 224, 224]
        # convert: 
        binary_mask = transform(np.squeeze(mask)).convert('1')
        binary_pred = transform(np.squeeze(pred)).convert('1')
        maskf = TF.to_tensor(binary_mask).view(-1)
        predf = TF.to_tensor(binary_pred).view(-1)

        return maskf, predf

    """ Method used to train the model with early stopping implementatinon. """
    def train(self):
        print('\nStarting training...\n')

        # keep track of average training loss
        avg_train_losses = []
        # keep track of average test(validation) loss
        avg_test_losses = []
        # keep track training performance for use in ablation studies
        my_dic_train_results = {}

        # trigger for earlystopping
        earlystopping = False

        # put the model in training mode
        self.net.train()

        # loop over the dataset multiple times
        for epoch in range(self.epochs):  
            # record the training and test
            # losses for each batch in this epoch
            train_losses = []
            test_losses = []

            # used for creating a terminal progress bar
            loop = tqdm(enumerate(self.train_loader), 
                        total=len(self.train_loader), leave=False)

            # loop over training data
            for batch, (images, targets, _) in loop:
                # put data on correct device
                images = images.to(self.device)
                targets = targets.to(self.device)

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                pred = self.net(images)
                
                # calculate the loss
                if self.args.loss == 'custom_loss':
                    loss = self.criterion(pred, targets, self.args.val_custom_loss)
                else:
                    loss = self.criterion(pred, targets)

                # backward pass: compute gradient 
                # of the loss with respect to model parameters
                loss.backward()

                if (batch % self.args.print_every == self.args.print_every - 1) or (batch == 0 and epoch == 0):
                    self.writer.add_figure('gradients_ebhi-seg', 
                                           add_gradient_hist(self.net), global_step=epoch * len(self.train_loader) + batch)
                
                # perform a single optimization step (parameter update)
                self.optimizer.step()

                # record training loss
                train_losses.append(loss.item())

                # used to check model improvements during the training
                self.check_results(batch, epoch)

                if batch % self.args.print_every == self.args.print_every - 1 or (batch == 0 and epoch == 0):
                    # save the model at the each (print_every - 1)
                    self.save_model()  

                    # validate the model at the each (print_every - 1)
                    if self.args.bs_test == 1:
                        (dc_class_test, jac_cust_class_test, jac_class_test, acc_class_test, 
                         prec_class_test, rec_class_test, f1s_class_test) = self.test(test_losses)
                    else:
                        self.test(test_losses)

                    # calculate average loss 
                    # (over a bacth because bs_train=4 and print_every=445)
                    batch_avg_train_loss = np.average(train_losses)
                    batch_avg_test_loss = np.average(test_losses)

                    avg_train_losses.append(batch_avg_train_loss)
                    avg_test_losses.append(batch_avg_test_loss)

                    print(f'\ntrain_loss: {batch_avg_train_loss:.5f} '
                          f'valid_loss: {batch_avg_test_loss:.5f}\n')

                    self.writer.add_scalar('batch_avg_train_loss', batch_avg_train_loss,
                                           global_step=epoch * len(self.train_loader) + batch)
                    self.writer.add_scalar('batch_avg_test_loss', batch_avg_test_loss,
                                           global_step=epoch * len(self.train_loader) + batch)
                    
                    """ # general dc in train/test
                    batch_avg_train_dc = np.average([1 - x for x in train_losses])
                    batch_avg_test_dc = np.average([1 - x for x in test_losses])
                    self.writer.add_scalar('batch_avg_train_dc', batch_avg_train_dc,
                                           global_step=epoch * len(self.train_loader) + batch)
                    self.writer.add_scalar('batch_avg_test_dc', batch_avg_test_dc,
                                           global_step=epoch * len(self.train_loader) + batch) """
                    
                    # saving some data in a dictionary (1):
                    # results are overwritten after each iteration since only the final 
                    # performance of the model is required for comparison between the ablated 
                    # and non-ablated models in the context of ablation studies                    
                    my_dic_train_results['epoch'] = str(epoch)
                    my_dic_train_results['global-step'] = str(epoch * len(self.train_loader) + batch)
                    my_dic_train_results['avg_train_losses'] = [str(x) for x in avg_train_losses]
                    my_dic_train_results['avg_test_losses'] = [str(x) for x in avg_test_losses]
                    
                    if self.args.bs_test == 1:
                        # computation of the corresponding average metric for each class
                        print(f'Dice for class {[np.average(x) for x in dc_class_test]}')
                        print(f'Jaccard index custom for class {[np.average(x) for x in jac_cust_class_test]}')
                        print(f'Binary Jaccard index for class {[np.average(x) for x in jac_class_test]}')
                        print(f'Binary Accuracy for class {[np.average(x) for x in acc_class_test]}')
                        print(f'Binary Precision for class {[np.average(x) for x in prec_class_test]}')
                        print(f'Binary Recall for class {[np.average(x) for x in rec_class_test]}')
                        print(f'Binary F1-score for class {[np.average(x) for x in f1s_class_test]}')

                        # log a Matplotlib Figure showing the metric for each class
                        self.writer.add_figure('accuracy_histo', add_metric_hist([np.average(x) for x in acc_class_test], 'Accuracy'), 
                                               global_step=epoch * len(self.train_loader) + batch)
                        self.writer.add_figure('precision_histo', add_metric_hist([np.average(x) for x in prec_class_test], 'Precision'), 
                                               global_step=epoch * len(self.train_loader) + batch)
                        self.writer.add_figure('recall_histo', add_metric_hist([np.average(x) for x in rec_class_test], 'Recall'), 
                                               global_step=epoch * len(self.train_loader) + batch)
                        
                        # saving some data in a dictionary (2)                 
                        my_dic_train_results['dc_class_test_mean'] = [str(np.average(x)) for x in dc_class_test]
                        my_dic_train_results['jac_cust_class_test_mean'] = [str(np.average(x)) for x in jac_cust_class_test]
                        my_dic_train_results['jac_class_test_mean'] = [str(np.average(x)) for x in jac_class_test]
                        my_dic_train_results['acc_class_test_mean'] = [str(np.average(x)) for x in acc_class_test]
                        my_dic_train_results['prec_class_test_mean'] = [str(np.average(x)) for x in prec_class_test]
                        my_dic_train_results['rec_class_test_mean'] = [str(np.average(x)) for x in rec_class_test]
                        my_dic_train_results['f1s_class_test_mean'] = [str(np.average(x)) for x in f1s_class_test]

                    print(f'\nGloabl-step: {epoch * len(self.train_loader) + batch}')

                    # clear statistics
                    train_losses = []
                    test_losses = []

                    # early stopping with a patience of 1 and a minimum of N epochs
                    if epoch > self.args.early_stopping:  
                        if avg_test_losses[-1] >= avg_test_losses[-2]:
                            print(f'\nEarly Stopping Triggered With Patience {self.args.early_stopping}')
                            # save before stop training
                            self.save_model()  
                            earlystopping = True
                    if earlystopping:
                        break

            if earlystopping:
                break

            # save at the end of each 
            # epoch only if earlystopping = False
            self.save_model()  

        # final analyses: kernel and activations
        # (at the end of the training process)
        self.kernel_analisys()           
        self.activation_analisys()

        # write all remaining data in the buffer
        self.writer.flush()
        # free up system resources used by the writer
        self.writer.close()

        # saving collected results in a json file
        self.save_json(my_dic_train_results)

        print('Training finished...\n')

    """ Method used to evaluate the model on the test set. """
    def test(self, test_losses):
        # put net into evaluation mode
        self.net.eval()  

        # no need to calculate the gradients for outputs
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

            # used for creating a terminal progress bar
            test_loop = tqdm(enumerate(self.test_loader),
                             total=len(self.test_loader), leave=False)

            for _, (test_images, test_targets, test_labels) in test_loop:
                test_images = test_images.to(self.device)
                test_targets = test_targets.to(self.device)
                test_pred = self.net(test_images.detach())

                # general loss
                if self.args.loss == 'custom_loss':
                    test_loss = self.criterion(test_pred, test_targets, self.args.val_custom_loss).item()
                else:
                    test_loss = self.criterion(test_pred, test_targets).item()

                test_losses.append(test_loss)

                if self.args.bs_test == 1:
                    # function that binarize an image and then convert it to a tensor
                    maskf, predf = self.binarization_tensor(test_targets, test_pred)

                    # per class metrics
                    dc_class_test[test_labels].append(1 - test_loss)
                    jac_cust_class_test[test_labels].append(1 - jac_loss(test_pred, test_targets).item())
                    jac_class_test[test_labels].append(binary_jac(maskf, predf))
                    acc_class_test[test_labels].append(binary_acc(maskf, predf))
                    prec_class_test[test_labels].append(binary_prec(maskf, predf))
                    rec_class_test[test_labels].append(binary_rec(maskf, predf))
                    f1s_class_test[test_labels].append(binary_f1s(maskf, predf))

        # put again the model in trainining-mode
        self.net.train()  

        if self.args.bs_test == 1:
            return dc_class_test, jac_cust_class_test, jac_class_test, acc_class_test, prec_class_test, rec_class_test, f1s_class_test

    """ Method used to visualize show some samples at
        the first batch of each epoch to check model improvements. """
    def check_results(self, batch, epoch):
        with torch.no_grad():
            if batch == 1:
                self.net.eval()

                (img, mask, label) = next(iter(self.test_loader))
                img = img.to(self.device)
                mask = mask.to(self.device)
                mask = mask[0]
                pred = self.net(img)

                self.writer.add_figure('check_results', plot_check_results(
                    img[0], mask, pred[0], label[0], self.args), global_step=epoch * len(self.train_loader) + batch)

                self.net.train()

    """ Method used to visualize CNN kernels. """
    def kernel_analisys(self):
        print('\nPerforming kernel analysis...\n')
        kernels_viewer(self.net, self.writer)

    """ Method used to visualize CNN activations. """
    def activation_analisys(self):
        print(f'\nPerforming kernel activations analysis...\n')
        # returns the next batch of data 
        # from the dataloader "self.test_loader"
        (img, _, _) = next(iter(self.test_loader))
        img = img.to(self.device)

        activations_viewer(self.net, self.writer, img)

    """ Same as quantify_change_nn_parameters
    def get_tensor_distance(self, mod1, mod2):
        p = mod1.weight.view(-1).detach().numpy()
        q = mod2.weight.view(-1).detach().numpy()
        
        return np.sqrt(np.sum(np.square(p - q)))
    """

    """ Method used to quantify the difference between the weights of a 
        neural network before and after training by subtracting the weight tensors 
        of the network before and after training and then computing the L2 norm. """
    def quantify_change_nn_parameters(self, tensor1, tensor2):
        diff = tensor2 - tensor1

        # p=2 "Frobenius norm"
        norm = torch.norm(diff, p=2)

        return norm

    """ Method used to run the comparison between 
        weights of the model before and after training. """
    def weights_pre_after_train_analysis(self):
        print(f'\nPerforming weights analysis pre-training vs. after '
              f'training to quantify neural network parameter changes...\n')

        model1_path = self.args.checkpoint_path + '/ebhi_seg_{}.pth'.format(self.args.model_name) 
        model2_path = self.args.checkpoint_path + '/' + self.args.model_name + '_before_training.pth'

        # there is no need to initialize the weights because the 
        # models that are to be loaded do not need to be trained
        self.args.weights_init = False

        if self.args.pretrained_net == True:
            self.net1 = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                       in_channels=3, out_channels=1, init_features=32, pretrained=True)
            self.net2 = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                       in_channels=3, out_channels=1, init_features=32, pretrained=True)
        elif self.args.arc_change_net == True:
            self.net1 = UNET(self.args, 3, 1, [int(f) for f in self.args.features])
            self.net2 = UNET(self.args, 3, 1, [int(f) for f in self.args.features])
        else:
            self.net1 = UNet(self.args)
            self.net2 = UNet(self.args)
            
        self.net1.load_state_dict(torch.load(model1_path, 
                                             map_location=torch.device(self.device)))
        self.net2.load_state_dict(torch.load(model2_path, 
                                             map_location=torch.device(self.device)))
        
        mod_name_list = []
        
        for (module_name1, module1), (module_name2, module2) in zip(self.net1.named_modules(), self.net2.named_modules()):
            if ((isinstance(module1, torch.nn.Conv2d) and ('bias' not in module_name1) 
                and isinstance(module2, torch.nn.Conv2d) and ('bias' not in module_name2)) or
                (isinstance(module1, torch.nn.Linear) and ('bias' not in module_name1) 
                and isinstance(module2, torch.nn.Linear) and ('bias' not in module_name2))):              
                
                # distance = self.get_tensor_distance(module1, module2)
                distance = self.quantify_change_nn_parameters(module1.weight.detach(), 
                                                              module2.weight.detach())
                mod_name_list.append((module_name1, distance))

                print(f'Distance between "{module_name1}" and "{module_name2}": {distance:.2f}')

        if self.args.single_mod == True or self.args.double_mod == True:
            # sort modules in descending order 
            # (related to variations during training-process)
            mod_name_list.sort(key=lambda tup: tup[1], reverse=True)

        mod_list = [t[0] for t in mod_name_list]

        return mod_list
        
    """ Method used to start ablation studies. """
    def start_ablation_study(self):
        print('\nStarting ablation studies...\n')

        ablationNn = AblationStudies(self.args, self.model_name, 
                                     self.train_loader, self.test_loader, 
                                     self.net, self.criterion, self.device, self.writer)
                
        # automatic modules-selection
        mod_name_list = self.weights_pre_after_train_analysis()

        # ablation study on the first or first two modules subjected-to 
        # more variations during training process (single_mode vs. double_mod)
        if self.args.single_mod == True:
            mod_name_list[:] = mod_name_list[0:1]
        elif self.args.double_mod == True:
            mod_name_list[:] = mod_name_list[0:2]
        # ablation study over all-modules of the model
        if self.args.global_ablation == True and self.args.grouped_pruning == True:
            ablationNn.iterative_pruning(grouped_pruning=True)
        elif self.args.global_ablation == True:
            ablationNn.iterative_pruning(grouped_pruning=False)
        # ablation study on all modules in mod_name_list
        elif self.args.selective_ablation == True:
            # # manual modules-selection
            # mod_name_list = ['downs.0.conv.0', 'ups.3.conv.2']
            ablationNn.selective_pruning(mod_name_list)
        # ablation study on all modules, but after each module 
        # the model is reloaded to evaluate the impact of pruning 
        # on each module independently from other modules
        elif self.args.all_one_by_one == True:
            for mod in mod_name_list:
                ablationNn.selective_pruning([mod])
                # reloading model after each unit-ablation study (network passed as reference)
                self.load_model(self.device)
                ablationNn = AblationStudies(self.args, self.model_name, self.train_loader, self.test_loader, 
                                             self.net, self.criterion, self.device, self.writer)
                
    """ Method used to plot the weights distribution using histograms. """
    def weights_distribution_analysis(self):
        print('\nPerforming weights analysis distribution...\n')
        plot_weights_distribution_histo(self.net, self.writer)
