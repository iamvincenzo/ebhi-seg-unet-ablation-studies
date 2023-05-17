import torch
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataloader_utils import class_dic, classes


""" Helper function used to set some style configurations. """
def set_default(figsize=(10, 10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)

""" Helper function used to plot an histogram to 
    represent the number of samples for each class. """
def add_sample_hist(n_sample_list, task):
    fig = plt.figure(figsize=(5, 5))
    plt.bar(np.arange(len(n_sample_list)), n_sample_list, lw=1, color='b')
    plt.xticks(range(0, len(n_sample_list), 1), classes, rotation=90)

    for i, v in enumerate(n_sample_list):
        plt.text(x=i-.26, y=v, s=v,
                 fontdict=dict(fontsize=10))

    plt.xlabel('Class')
    plt.ylabel('Number of elements')

    plt.title('Number of elements per class: ' + task)

    plt.tight_layout()

    # plt.show()

    return fig

""" Helper function used to plot the gradients 
    histogram for each layer in the network and its values. """
def add_gradient_hist(net):
    ave_grads = []
    layers = []
    for n, p in net.named_parameters():
        if ("bias" not in n):
            layers.append(n)
            if p.requires_grad:  # indicates whether a variable is trainable
                ave_grad = np.abs(p.grad.clone().detach().cpu().numpy()).mean()
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
    
    # plt.show()

    return fig

""" Helper function used to plot the performance of 
    the model for each class according to a specific metric. """
def add_metric_hist(metr_list, metr):
    fig = plt.figure(figsize=(5, 5))
    plt.bar(np.arange(len(metr_list)), metr_list, lw=1, color='b')
    plt.xticks(range(0, len(metr_list), 1), classes, rotation=90)

    for i, v in enumerate(metr_list):
        plt.text(x=i-.26, y=v, s='{:.3f}'.format(v),
                 fontdict=dict(fontsize=10))

    plt.xlabel('Class')
    plt.ylabel(metr)

    plt.title('Classes ' + metr)

    plt.tight_layout()

    # plt.show()

    return fig

""" Helper function used to denormalize an image. """
def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

""" Helper function used to 
    plot the image of a dataset-sample. """
def plot_samples(images, mask, labels, args):
    fig = plt.figure(figsize=(12, 12))
    rows, columns, j = 2, args.bs_train, 0
    for i in range(1, columns*rows + 1):
        fig.add_subplot(rows, columns, i)
        # row-1 contains images
        if i <= args.bs_train:
            if args.norm_input:
                plt.imshow((denorm((images[j].cpu())).numpy()).transpose(1, 2, 0))
            else:
                                                    # from [3, 244, 244] to [244, 244, 3]
                plt.imshow((images[j].cpu().numpy()).transpose(1, 2, 0))
            plt.title(f'{list(class_dic.keys())[list(class_dic.values()).index(labels[j].item())]}') # class of the sample
        # row-2 contains masks
        else:
            plt.imshow((mask[j].cpu().numpy()).transpose(1, 2, 0).squeeze(axis=2))
        
        # reinitialize the index used 
        # to iterate over tensors (row=2)
        if j < args.bs_train:
            j += 1
        if j == args.bs_train:
            j = 0

    # plt.show()

    return fig

""" Helper function used to plot the 
    image, the mask and the model prediction. """
def plot_check_results(img, mask, pred, label, args):
    fig = plt.figure(figsize=(12, 12))
    plt.subplot(1, 3, 1)
    if args.norm_input:
        plt.imshow(np.squeeze((denorm(img.cpu())).numpy()).transpose(1, 2, 0))
    else:
        plt.imshow(np.squeeze(img.cpu().numpy()).transpose(1, 2, 0))
                                  # inverse-search: from number (value) to label (key) 
    plt.title(f'Original Image - {list(class_dic.keys())[list(class_dic.values()).index(label.item())]}')
    plt.subplot(1, 3, 2)
    plt.imshow((mask.cpu().numpy()).transpose(1, 2, 0).squeeze(axis=2))
    plt.title('Original Mask')
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(pred.cpu()) > .5)
    plt.title('Prediction')

    # plt.show()

    return fig

""" Helper function used to visualize CNN kernels. """
def kernels_viewer(model, wrt):
    layers_list = []
    j = 0

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            kernels = module.weight.detach().clone().cpu()

            if kernels.shape[0] > 1:
                layers_list.append(module_name)

                title = module_name + ' ' + str(kernels.shape)

                if j == 0:
                    # used to get only the first cast_dim filters
                    cast_dim = kernels.shape[0] 
                    rows = 4
                    cols = kernels.shape[0] // rows

                n, c, w, h = kernels.shape

                # es. [32, 3, 3, 3] --> [96, 1, 3, 3] 
                # --> kernels[i].permute(1, 2, 0) --> img(3, 3, 1) = (w, h, c)
                kernels = kernels.view(n*c, -1, w, h)

                if kernels.shape[0] > cast_dim:
                    # [96, 1, 3, 3] --> [64, 1, 3, 3] (cast_dim=64)
                    kernels = kernels[:cast_dim] 

                # normalize to (0, 1) range so that matplotlib can plot them
                kernels = kernels - kernels.min()
                kernels = kernels / kernels.max()

                filter_img_fig = plt.figure(figsize=(15, len(layers_list)))
                
                for i in range(kernels.shape[0]):
                    plt.subplot(rows, cols, i + 1)
                    # change ordering since matplotlib requires images to be (H, W, C)
                    plt.imshow(kernels[i].permute(1, 2, 0)) 
                    plt.suptitle(title)
                    plt.axis('off')

                # plt.show()

                wrt.add_figure('filter_img_grid_' + str(j), filter_img_fig)
                j += 1

# Use hooks: 
# "hooks" are a mechanism that allows intercepting and recording 
# activities within a neural network module during data processing
conv_output = []

""" Helper function used to visualize CNN activations. """
def append_conv(module, input, output):
    # append all the conv layers and their respective wights to the list
    conv_output.append(output.detach().cpu()) 

""" Helper function used to visualize CNN activations. """
def activations_viewer(net, wrt, img):
    j = 0

    for module_name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # hooks executed during the forward 
            # pass of data through a neural network module
            module.register_forward_hook(append_conv)

    out = net(img)

    # for c_out in conv_output:
    #     print(f'c_out-size: {c_out.size()}')
    # print('\n\n')

    for num_layer in range(len(conv_output)):
        act_img_fig = plt.figure(figsize=(30, 30))
        layer_viz = conv_output[num_layer][0, :64, :, :]
        layer_viz = layer_viz.data
        # print(f'layer_viz: {layer_viz.size()}')
        for i, filter in enumerate(layer_viz):
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter)
            plt.title(f'{filter.shape[0]} x {filter.shape[1]}')
            plt.axis("off")

        plt.show()

        wrt.add_figure('activations_img_grid_' + str(j), act_img_fig)
        j += 1
        
""" Helper function used to plot histogram of weights values. """
def plot_weights_distribution_histo(model, writer, bins = 100):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and ('bias' not in module_name):
            w = module.weight.detach().view(-1)
            hist = torch.histc(w, bins=bins, min=torch.min(w).item(), max=torch.max(w).item(), out=None)
            fig = plt.figure(figsize=(5, 5))
            plt.bar(range(bins), hist, align='center', color=['forestgreen'])
            plt.xlabel('Bins')
            plt.ylabel('Frequency')
            title = f'Weights distribution of {module_name} module '
            plt.title(title, fontsize=18)
            plt.show()
            writer.add_figure(title, fig)

    writer.close()

""" Helper function used to plot some metrics. """
def bar_plotting(l0, l1, l2, metric):    
    if metric == 'acc_class_test_mean':
        k0, k1 = 'train_acc', 'abl_acc'
    elif metric == 'prec_class_test_mean':
        k0, k1 = 'train_prec', 'abl_prec'
    elif metric == 'rec_class_test_mean':
        k0, k1 = 'train_rec', 'abl_rec'
    elif metric == 'dc_class_test_mean':
        k0, k1 = 'train_dc', 'abl_dc'

    df = pd.DataFrame({k0: l0, k1: l1, 'drop': l2}, index=classes)

    cmap = cm.get_cmap('Set2') # Colour map (there are many others)
    ax = df.plot.bar(rot=0, cmap=cmap, edgecolor='None',  figsize=(12, 10))
    
    for p in ax.patches:
        ax.annotate("{:.3f}".format(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    return ax

""" Helper function used to plot area graph. """
def area_plotting(l0, l1, l2, metric, stckd=True, subplts=True):    
    if metric == 'acc_class_test_mean':
        k0, k1 = 'train_acc', 'abl_acc'
    elif metric == 'prec_class_test_mean':
        k0, k1 = 'train_prec', 'abl_prec'
    elif metric == 'rec_class_test_mean':
        k0, k1 = 'train_rec', 'abl_rec'
    elif metric == 'dc_class_test_mean':
        k0, k1 = 'train_dc', 'abl_dc'

    df = pd.DataFrame({k0: l0, k1: l1, 'drop': l2}, index=classes)

    cmap = cm.get_cmap('Set2') # Colour map (there are many others)
    ax = df.plot.area(stacked=stckd, cmap=cmap, subplots=subplts, figsize=(12, 10))

    return ax

""" Helper function used to plot line graph. """
def line_plotting(l0, l1, metric):    
    fig, axes = plt.subplots(2, 1)

    if metric == 'acc_class_test_mean':
        k0, k1 = 'train_acc', 'abl_acc'
    elif metric == 'prec_class_test_mean':
        k0, k1 = 'train_prec', 'abl_prec'
    elif metric == 'rec_class_test_mean':
        k0, k1 = 'train_rec', 'abl_rec'
    elif metric == 'dc_class_test_mean':
        k0, k1 = 'train_dc', 'abl_dc'

    df = pd.DataFrame({k0: l0, k1: l1}, index=classes)

    # Colour map (there are many others)
    cmap = cm.get_cmap('Set2') 
    df.plot.line(cmap=cmap, ax=axes[0], figsize=(12, 5))

    my_array = np.array([l0, l1])
    df = pd.DataFrame(my_array, columns=classes)
    df.plot.line(cmap=cmap, ax=axes[1], figsize=(12, 10))
    
    return fig
