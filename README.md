# Ablation-Studies on UNet Architecture trained on EBHI-Seg dataset (ENG)

The objective of this deep-learning task is to conduct ablation studies on the UNet network architecture, which has been trained on the EBHI-Seg dataset. The project consists of the following Python modules:

- main_test.py: This module is responsible for launching the simulation. It utilizes other modules, including:
  (i) Modules for handling training/validation data (dataloader generation, balancing, augmentation).
  (ii) Modules for training and validation.
  (iii) Modules for conducting ablation studies on the trained model.

- data_cleaning.py: This module is responsible for automatic data cleaning. Specifically, during data analysis, it was observed that one class contained two images without the corresponding binary mask, so those images were removed. It also provides a function to delete images generated by the "data_augmentation.py" module.

- dataloader_utils.py: This module is responsible for generating well-balanced dataloaders for training and validation.

- data_augmentation.py: This module provides automated support for the data augmentation process.

- data_balancing.py: This module is responsible for creating a training dataloader in which examples from all classes are presented an equal number of times, ensuring fair learning of the classes by the network.

- metrics.py: This module contains various metrics and loss functions used during training and validation.

- model.py and arc_change_net.py: These modules contain an implementation of the UNet network architecture. In particular, the second module offers a more customizable architecture in terms of depth (number of network layers), width (number of filters i.e., neurons per layer), use of normalization techniques, and more.

- plotting_utils.py: This module contains several useful functions for plotting metrics, results, kernels, activations, and more.

- solver.py: This module includes various methods for training, validation, and conducting ablation studies after training the model. It also provides functionality for saving and loading the model, visualizing model weights, activations, and kernels.

- ablation_studies.py: This module is responsible for executing ablation studies based on sensitivity analysis and pruning of network modules.

<!-- - results_analysis.py: This module (ad-hoc) is specifically created to obtain the best network configurations based on the analysis of statistics collected during training. Additionally, it allows for visualizing the results of ablation studies, based on the analysis of statistics collected during the execution of ablation studies.-->

The results are contained within the "final_consideration.pdf" file.

**The dataset can be downloaded at:** https://paperswithcode.com/dataset/ebhi-seg

**Other interesting resources:**

- **U-Net: Convolutional Networks for Biomedical Image Segmentation:** https://arxiv.org/abs/1505.04597
- **EBHI-Seg: A Novel Enteroscope Biopsy Histopathological Haematoxylin and Eosin Image Dataset for Image Segmentation Tasks:** https://arxiv.org/abs/2212.00532
- **Ablation Studies in Artificial Neural Networks:** https://arxiv.org/abs/1901.08644


## The parameters that can be provided through the command line and allow customization of the execution are:


| Key                   | Value Description                                          |
|-----------------------|------------------------------------------------------------|
| run_name              | The name assigned to the current run                        |
| model_name            | The name of the model to be saved or loaded                 |
| epochs                | The total number of training epochs                         |
| bs_train              | The batch size for training data                            |
| bs_test               | The batch size for test data                                |
| workers               | The number of workers in the data loader                    |
| print_every           | The frequency of printing losses during training            |
| random_seed           | The random seed used to ensure reproducibility              |
| lr                    | The learning rate for optimization                          |
| loss                  | The loss function used for model optimization               |
| val_custom_loss       | The weight of the jac_loss in the overall loss function     |
| opt                   | The optimizer used for training                             |
| early_stopping        | The threshold for early stopping during training            |
| resume_train          | Determines whether to load the model from a checkpoint      |
| use_batch_norm        | Indicates whether to use batch normalization layers in each conv layer |
| use_double_batch_norm | Indicates whether to use 2 batch normalization layers in each conv layer |
| use_inst_norm         | Indicates whether to use instance normalization layers in each conv layer |
| use_double_inst_norm  | Indicates whether to use 2 instance normalization layers in each conv layer |
| weights_init          | Determines whether to use weights initialization            |
| th                    | The threshold used to split the dataset into train and test subsets |
| dataset_path          | The path to save or retrieve the dataset                    |
| checkpoint_path       | The path to save the trained model                          |
| pretrained_net        | Indicates whether to load a pretrained model on BRAIN-MRI   |
| arc_change_net        | Indicates whether to load a customizable model |
| features              | A list of feature values (number of filters i.e., neurons in each layer)     |
| norm_input            | Indicates whether to normalize the input data               |
| apply_transformations | Indicates whether to apply transformations to images and corresponding masks |
| dataset_aug           | Determines the type of data augmentation applied to each class |
| balanced_trainset     | Generates a well-balanced train_loader for training         |
| global_ablation       | Initiates an ablation study using global_unstructured or l1_unstructured |
| grouped_pruning       | Initiates an ablation study using global_unstructured       |
| all_one_by_one        | Initiates an ablation study using random_structured or random_unstructured |
| selective_ablation    | Initiates an ablation study using random_structured or random_unstructured |
| single_mod            | Initiates an ablation study with only one module            |
| double_mod            | Initiates an ablation study with two modules                |
| random_structured     | Initiates an ablation study using random_structured         |
| random_unstructured   | Initiates an ablation study using random_unstructured       |
| conv2d_prune_amount   | The amount of pruning applied to conv2d layers              |
| linear_prune_amount   | The amount of pruning applied to linear layers              |
| num_iterations        | The number of iterations for the pruning process            |
| weights_distr_histo   | Plots the histogram of weights distribution                 |
| plt_weights_distr     | Plots the filters as images to visualize CNN kernels        |


# Studi di ablazione dell'architettura di rete UNet addestrata sul dataset EBHI-Seg (ITA)

Il task di deep-learning in questione consiste nell'esecuzione di studi di ablazione sull'architettura di rete UNet addestrata sul dataset EBHI-Seg. Pertanto, i moduli Python che compongono il progetto sono:

- main_test.py: è il modulo responsabile del lancio della simulazione. Al suo interno, vengono utilizzati altri moduli, tra cui:
  (i) moduli per la gestione dei dati di training/validation (generazione dataloader, bilanciamento, aumento).
  (ii) moduli per il training e la validazione.
  (iii) moduli per eseguire gli studi di ablazione sul modello addestrato.

- data_cleaning.py: è il modulo responsabile per la pulizia automatica dei dati. In particolare, durante l'analisi dei dati, è stato osservato che una classe conteneva due immagini senza la corrispondente maschera binaria, quindi sono state rimosse. Inoltre, fornisce una funzione per eliminare le immagini generate dal modulo "data_augmentation.py".

- dataloader_utils.py: è il modulo responsabile per la generazione di dataloader ben bilanciati per il training e la validazione.

- data_augmentation.py: è il modulo respondabile per supporto automatico al processo di data-augmentation.

- data_balancing.py: è il modulo responsabile per la creazione di un dataloader di training in cui gli esempi di tutte le classi vengono presentati lo stesso numero di volte, garantendo così un apprendimento equo delle classi da parte della rete.

- metrics.py: è un modulo che contiene diverse metriche e funzioni di loss utilizzate durante il training e la validazione.

- model.py e arc_change_net.py: sono moduli che contengono un'implementazione dell'architettura di rete UNet. In particolare, il secondo modulo offre un'architettura più personalizzabile in termini di profondità (numero di layer della rete), ampiezza (numero di neuroni o filtri per ogni layer), uso di tecniche di normalizzazione e altro.

- plotting_utils.py: contiene alcune funzioni utili per il plotting di metriche, kernel, attivazioni e altro ancora.

- solver.py: è il modulo che include vari metodi per il training, la validazione e l'esecuzione degli studi di ablazione dopo l'addestramento del modello. Fornisce anche funzionalità per salvare e caricare il modello, visualizzare i pesi del modello, le attivazioni e i kernel.

- ablation_studies.py: è il modulo responsabile per l'esecuzione degli studi di ablazione basati sull'analisi di sensitività e il pruning dei moduli della rete.

<!-- - results_analysis.py: è un modulo (ad-hoc) appositamente creato per ottenere le migliori configurazioni di rete basandosi sull'analisi delle statistiche raccolte durante il training. Inoltre, consente di visualizzare i risultati degli studi di ablazione, basandosi sempre sull'analisi delle statistiche raccolte durante l'esecuzione degli studi di ablazione. -->
