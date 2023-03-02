import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataloader_utils import class_dic

classes = ['Normal', 'Polyp', 'Low-grade IN',
           'High-grade IN', 'Adenocarcinoma', 'Serrated adenoma']


""" Helper function used to set some style
    configurations. """
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


""" Helper function used to plot the gradients list for each layer in the network 
    and its values. """
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
    # plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90)
    # plt.xlim(left=0, right=len(ave_grads))
    # zoom in on the lower gradient regions
    # plt.ylim(bottom=-0.001, top=np.max(ave_grads) / 2)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    # plt.grid(True)
    plt.legend([Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    
    # plt.show()

    return fig


""" Helper function used to plot the performance of the model
    for each class according to a specific metric. """
def add_metric_hist(metr_list, metr):
    fig = plt.figure(figsize=(5, 5))
    plt.bar(np.arange(len(metr_list)), metr_list, lw=1, color='b')
    plt.xticks(range(0, len(metr_list), 1), classes, rotation=90)
    plt.ylim(bottom=0.0, top=1.0)

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


""" Helper function used to plot the image
    of a dataset-sample. """
def plot_samples(images, mask, labels, args):
    fig = plt.figure(figsize=(12, 12))
    rows, columns, j = 2, args.bs_train, 0
    for i in range(1, columns*rows + 1):
        fig.add_subplot(rows, columns, i)
        if i <= args.bs_train:
            if args.norm_input:
                plt.imshow(
                    (denorm((images[j].cpu())).numpy()).transpose(1, 2, 0))
            else:
                plt.imshow((images[j].cpu().numpy()).transpose(1, 2, 0)) # from [3, 244, 244] to [244, 244, 3]
            plt.title(f'{list(class_dic.keys())[list(class_dic.values()).index(labels[j].item())]}') # class of the sample
        else:
            plt.imshow((mask[j].cpu().numpy()).transpose(1, 2, 0).squeeze(axis=2))

        if j < args.bs_train:
            j += 1
        if j == args.bs_train:
            j = 0

    # plt.show()

    return fig


""" Helper function used to plot the image
    the mask and the model prediction. """
def plot_check_results(img, mask, pred, label, args):
    fig = plt.figure(figsize=(12, 12))
    plt.subplot(1, 3, 1)
    if args.norm_input:
        plt.imshow(np.squeeze((denorm(img.cpu())).numpy()).transpose(1, 2, 0))
    else:
        plt.imshow(np.squeeze(img.cpu().numpy()).transpose(1, 2, 0))
    plt.title(f'Original Image - {list(class_dic.keys())[list(class_dic.values()).index(label.item())]}')
    plt.subplot(1, 3, 2)
    plt.imshow((mask.cpu().numpy()).transpose(1, 2, 0).squeeze(axis=2))
    plt.title('Original Mask')
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(pred.cpu()) > .5)
    plt.title('Prediction')

    # plt.show()

    return fig


""" Helper function used to visualize 
    CNN kernels. """
def kernels_viewer(layers_list, out_l, wrt):
    index_list = [0, 2]
    j = 0

    for layer in layers_list:
        for k in index_list:
            # get the kernels from the first layer as per the name of the layer
            if layer != out_l:
                kernels = layer[k].weight.detach().clone().cpu()
            else:
                kernels = layer.weight.detach().clone().cpu()

            ## check size for sanity check
            # print(kernels.size())

            n, c, w, h = kernels.shape
            row_num = 8

            kernels = kernels.view(n*c, -1, w, h)

            rows = np.min((kernels.shape[0] // row_num + 1, 16))

            if kernels.shape[0] > 64:
                kernels = kernels[:64]
            # normalize to (0,1) range so that matplotlib
            # can plot them
            kernels = kernels - kernels.min()
            kernels = kernels / kernels.max()

            # filter_img = torchvision.utils.make_grid(kernels, nrow=rows)

            filter_img_fig = plt.figure(figsize=(15, len(layers_list)))
            for i in range(kernels.shape[0]):
                plt.subplot(row_num, rows, i + 1)
                # change ordering since matplotlib requires images to be (H, W, C)
                plt.imshow(kernels[i].permute(1, 2, 0))
                plt.axis('off')

            # plt.show()

            wrt.add_figure('filter_img_grid_' + str(j), filter_img_fig)
            j += 1


# Use HOOKS
conv_output = []

def append_conv(module, input, output):
    conv_output.append(output.detach().cpu()) # append all the conv layers and their respective wights to the list

""" Helper function used to visualize 
    CNN activations. """
def activations_viewer(layers_list, net, wrt, img, out_l):
    index_list = [0, 2]
    j = 0

    for layer in layers_list:
        for k in index_list:
            # get the kernels from the first layer as per the name of the layer
            if layer != out_l:
                layer[k].register_forward_hook(append_conv)
            else:
                layer.register_forward_hook(append_conv)

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

        # plt.show()

        wrt.add_figure('activations_img_grid_' + str(j), act_img_fig)
        j += 1
