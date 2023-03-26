import os
import json
import torch
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from model import UNet
from arc_change_net import UNET
from ablation_studies import AblationStudies
from metrics import dc_loss, jac_loss, custom_loss, binary_jac, binary_acc, binary_prec, binary_rec, binary_f1s
from plotting_utils import set_default, add_gradient_hist, add_metric_hist, plot_check_results, kernels_viewer, activations_viewer


""" Solver for training and testing. """
class Solver(object):
    """ Initialize configurations. """
    def __init__(self, train_loader, test_loader, device, writer, args):
        self.args = args
        self.model_name = 'ebhi_seg_{}.pth'.format(self.args.model_name)

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

        # define the model
        self.net = model.to(device)

        # load a pretrained model
        if (self.args.resume_train == True or self.args.global_ablation == True 
            or self.args.selective_ablation == True or self.args.all_one_by_one == True):
            self.load_model(device)

        # define Loss function
        if self.args.loss == 'dc_loss':
            self.criterion = dc_loss
            print(f'\nDC_loss selected!\n')

        elif self.args.loss == 'jac_loss':
            self.criterion = jac_loss
            print(f'\nJAC_loss selected!\n')

        elif (self.args.loss == 'bcewl_loss' 
              and self.args.arc_change_net == True):
            self.criterion = nn.BCEWithLogitsLoss()
            print(f'\nBCEWithLogitsLoss selected!\n')

        elif self.args.loss == 'custom_loss':
            self.criterion = custom_loss
            print(f'\ncustom_loss selected!\n')


        # choose optimizer
        if self.args.opt == "SGD":
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.opt == "Adam":
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=self.args.lr, betas=(0.9, 0.999))


        self.epochs = self.args.epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.writer = writer


        # visualize the model we built on tensorboard
        images, _, _ = next(iter(self.train_loader)) # images = images.to(device) ???
        self.writer.add_graph(self.net, images.to(self.device))
        self.writer.close()

        set_default() # setting fig-style


    """ Helper function used to save the model. """
    def save_model(self):
        # if you want to save the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        torch.save(self.net.state_dict(), check_path)
        print('\nModel saved!\n')


    """ Helper function used to load the model. """
    def load_model(self, device):
        # function to load the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(check_path, 
                                            map_location=torch.device(device)))
        print('\nModel loaded!\n')


    """ Helper function used to save collected training-statistics. """
    def save_json(self, file):
        with open('./statistics/my_dic_train_results_' + self.args.model_name + 
                  '_' + datetime.datetime.now().strftime('%d%m%Y-%H%M%S') + '.json', 'w') as f:
            json.dump(file, f)


    """ Helper function used to binarize a tensor (mask)
        in order to compute binary accuracy, precision, recall, f1-score. """
    def binarization_tensor(self, mask, pred):
        transform = T.ToPILImage()
        binary_mask = transform(np.squeeze(mask)).convert('1')
        binary_pred = transform(np.squeeze(pred)).convert('1')
        maskf = TF.to_tensor(binary_mask).view(-1)
        predf = TF.to_tensor(binary_pred).view(-1)

        return maskf, predf


    """ Helper function used to train the model with early stopping implementatinon. """
    def train(self):
        print('\nStarting training...\n')

        # keep track of average training and test losses for each epoch
        avg_train_losses = []
        avg_test_losses = []
        my_dic_train_results = {}

        # trigger for earlystopping
        earlystopping = False

        self.net.train()

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            # record the training and test losses for each batch in this epoch
            train_losses = []
            test_losses = []

            loop = tqdm(enumerate(self.train_loader),
                        total=len(self.train_loader), leave=False)

            for batch, (images, targets, _) in loop:
                images = images.to(self.device)
                targets = targets.to(self.device)  # the ground truth mask

                # zero the parameter gradients
                self.optimizer.zero_grad()  # self.net.zero_grad() ???

                # forward + backward + optimize
                pred = self.net(images)
                
                if self.args.loss == 'custom_loss':
                    loss = self.criterion(pred, targets, self.args.val_custom_loss)
                else:
                    loss = self.criterion(pred, targets)

                loss.backward()

                # plot gradient histogram distribution ????
                if (batch % self.args.print_every == self.args.print_every - 1) or (batch == 0 and epoch == 0):
                    self.writer.add_figure('gradients_ebhi-seg', add_gradient_hist(
                        self.net), global_step=epoch * len(self.train_loader) + batch)

                self.optimizer.step()

                train_losses.append(loss.item())

                # used to check model improvement
                self.check_results(batch, epoch)

                if batch % self.args.print_every == self.args.print_every - 1 or (batch == 0 and epoch == 0):
                    self.save_model()  # save at the each (print_every - 1)

                    # Test model
                    if self.args.bs_test == 1:
                        dc_class_test, jac_cust_class_test, jac_class_test, acc_class_test, prec_class_test, rec_class_test, f1s_class_test = self.test(
                            test_losses)
                    else:
                        self.test(test_losses)

                    batch_avg_train_loss = np.average(train_losses)
                    batch_avg_test_loss = np.average(test_losses)

                    avg_train_losses.append(batch_avg_train_loss)
                    avg_test_losses.append(batch_avg_test_loss)

                    # general dc in train/test
                    batch_avg_train_dc = np.average(
                        [1 - x for x in train_losses])
                    batch_avg_test_dc = np.average(
                        [1 - x for x in test_losses])

                    print(f'\ntrain_loss: {batch_avg_train_loss:.5f} ' +
                          f'valid_loss: {batch_avg_test_loss:.5f}\n')

                    self.writer.add_scalar('batch_avg_train_loss', batch_avg_train_loss,
                                           global_step=epoch * len(self.train_loader) + batch)
                    self.writer.add_scalar('batch_avg_test_loss', batch_avg_test_loss,
                                           global_step=epoch * len(self.train_loader) + batch)

                    self.writer.add_scalar('batch_avg_train_dc', batch_avg_train_dc,
                                           global_step=epoch * len(self.train_loader) + batch)
                    self.writer.add_scalar('batch_avg_test_dc', batch_avg_test_dc,
                                           global_step=epoch * len(self.train_loader) + batch)
                    
                    
                    """ Saving some data in a dictionary (1). """                    
                    my_dic_train_results['epoch'] = str(epoch)
                    my_dic_train_results['global-step'] = str(epoch * len(self.train_loader) + batch)
                    my_dic_train_results['avg_train_losses'] = [str(x) for x in avg_train_losses]
                    my_dic_train_results['avg_test_losses'] = [str(x) for x in avg_test_losses]
                    
                    
                    if self.args.bs_test == 1:
                        print(f'Dice for class {[np.average(x) for x in dc_class_test]}')
                        print(f'Jaccard index custom for class {[np.average(x) for x in jac_cust_class_test]}')
                        print(f'Binary Jaccard index for class {[np.average(x) for x in jac_class_test]}')
                        print(f'Binary Accuracy for class {[np.average(x) for x in acc_class_test]}')
                        print(f'Binary Precision for class {[np.average(x) for x in prec_class_test]}')
                        print(f'Binary Recall for class {[np.average(x) for x in rec_class_test]}')
                        print(f'Binary F1-score for class {[np.average(x) for x in f1s_class_test]}')

                        # log a Matplotlib Figure showing the metric for each class
                        self.writer.add_figure('accuracy_histo', add_metric_hist([np.average(
                            x) for x in acc_class_test], 'Accuracy'), global_step=epoch * len(self.train_loader) + batch)
                        self.writer.add_figure('precision_histo', add_metric_hist([np.average(
                            x) for x in prec_class_test], 'Precision'), global_step=epoch * len(self.train_loader) + batch)
                        self.writer.add_figure('recall_histo', add_metric_hist([np.average(
                            x) for x in rec_class_test], 'Recall'), global_step=epoch * len(self.train_loader) + batch)
                        
                        """ Saving some data in a dictionary (2). """                    
                        my_dic_train_results['dc_class_test_mean'] = [str(np.average(x)) for x in dc_class_test]
                        my_dic_train_results['jac_cust_class_test_mean'] = [str(np.average(x)) for x in jac_cust_class_test]
                        my_dic_train_results['jac_class_test_mean'] = [str(np.average(x)) for x in jac_class_test]
                        my_dic_train_results['acc_class_test_mean'] = [str(np.average(x)) for x in acc_class_test]
                        my_dic_train_results['prec_class_test_mean'] = [str(np.average(x)) for x in prec_class_test]
                        my_dic_train_results['rec_class_test_mean'] = [str(np.average(x)) for x in rec_class_test]
                        my_dic_train_results['f1s_class_test_mean'] = [str(np.average(x)) for x in f1s_class_test]

                    print(f'\nGloabl-step: {epoch * len(self.train_loader) + batch}')

                    train_losses = []
                    test_losses = []

                    if epoch > self.args.early_stopping:  # Early stopping with a patience of 1 and a minimum of N epochs
                        if avg_test_losses[-1] >= avg_test_losses[-2]:
                            print('\nEarly Stopping Triggered With Patience 1')
                            self.save_model()  # save before stop training
                            earlystopping = True
                    if earlystopping:
                        break

            if earlystopping:
                break

            self.save_model()  # save at the end of each epoch

        # final analyses (at the end of the training process)
        self.kernel_analisys()           
        self.activation_analisys()

        self.writer.flush()
        self.writer.close()
        print('Finished Training!\n')

        """ Saving collected results in a file. """
        self.save_json(my_dic_train_results)


    """ Helper function used to evaluate the model on the test set. """
    def test(self, test_losses):
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

        self.net.train()  # put again the model in trainining-mode

        if self.args.bs_test == 1:
            return dc_class_test, jac_cust_class_test, jac_class_test, acc_class_test, prec_class_test, rec_class_test, f1s_class_test


    """ Helper function used to visualize show some samples at
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


    """ Helper function used to visualize CNN kernels. """
    def kernel_analisys(self):
        print('\nPerforming kernel analysis...\n')

        kernels_viewer(self.net, self.writer)


    """ Helper function used to visualize CNN activations. """
    def activation_analisys(self):
        print(f'\nPerforming kernel activations analysis...\n')

        (img, _, _) = next(iter(self.test_loader))
        img = img.to(self.device)

        activations_viewer(self.net, self.writer, img)


    ############################################## PREVIEW ##################################################

    """ plot_kernels
    def plot_kernels(self, tensor):
        import matplotlib.pyplot as  plt
        
        if not tensor.ndim==4:
            raise Exception("assumes a 4D tensor")
        if not tensor.shape[-1]==3:
            raise Exception("last dim needs to be 3 to plot")
        
        num_cols=tensor.shape[0]
        num_kernels = tensor.shape[0]
        num_rows = 1 #+ num_kernels // num_cols

        print(num_cols, num_kernels, num_rows)

        fig = plt.figure(figsize=(num_cols, num_rows))
        for i in range(tensor.shape[0]):
            ax1 = fig.add_subplot(num_rows,num_cols,i+1)
            ax1.imshow(tensor[i])
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

        # plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.show()
    """

    def get_tensor_distance(self, mod1, mod2):
        p = mod1.weight.view(-1).detach().numpy()
        q = mod2.weight.view(-1).detach().numpy()
        
        return np.sqrt(np.sum(np.square(p - q)))


    def weights_distribution_analysis(self):
        print('\nPerforming weights analysis distribution...\n')

        model1_path = self.args.checkpoint_path + '/ebhi_seg_{}.pth'.format(self.args.model_name) 
        model2_path = self.args.checkpoint_path + '/' + self.args.model_name + '_before_training.pth'


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

                """ Plotting weights
                tensor1 = module1.weight.detach()
                tensor2 = module2.weight.detach()

                if tensor1.shape[0] > 64:
                    tensor1 = tensor1[:64]

                if tensor2.shape[0] > 64:
                    tensor2 = tensor2[:64]
                
                tensor1 = tensor1 - tensor1.min()
                tensor1 = tensor1 / tensor1.max()

                tensor2 = tensor2 - tensor2.min()
                tensor2 = tensor2 / tensor2.max()
                
                tensor1 = tensor1.numpy()
                tensor2 = tensor2.numpy()
                
                self.plot_kernels(tensor1)
                self.plot_kernels(tensor2)
                """
                
                distance = self.get_tensor_distance(module1, module2)
                mod_name_list.append((module_name1, distance))

                print(f'Distance between "{module_name1}" and "{module_name2}": {distance:.2f}')


        mod_name_list.sort(key=lambda tup: tup[1], reverse=True)

        mod_list = [t[0] for t in mod_name_list]

        return mod_list
        
    #########################################################################################################


    """ Helper function used to start some ablation studies. """
    def start_ablation_study(self):
        print('\nStarting ablation studies...\n')
        
        ablationNn = AblationStudies(self.args, self.model_name, self.train_loader, self.test_loader, 
                                     self.net, self.criterion, self.device, self.writer)
        
        mod_name_list = self.weights_distribution_analysis()


        if self.args.single_mod == True:
            mod_name_list[:] = mod_name_list[0:1]
        elif self.args.double_mod == True:
            mod_name_list[:] = mod_name_list[0:2]
        

        if self.args.global_ablation == True and self.args.grouped_pruning == True:
            ablationNn.iterative_pruning_finetuning(True)
        elif self.args.global_ablation == True:
            ablationNn.iterative_pruning_finetuning()
        elif self.args.selective_ablation == True:
            ablationNn.selective_pruning(mod_name_list)
        elif self.args.all_one_by_one == True:
            for mod in mod_name_list:
                ablationNn.selective_pruning([mod])



    