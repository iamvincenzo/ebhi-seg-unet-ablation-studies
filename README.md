# Ablation studies on UNet architecture trained on EBHI-Seg dataset (ENG)

The objective of this deep-learning task is to conduct ablation studies on the UNet network architecture, which has been trained on the EBHI-Seg dataset. The project consists of the following Python modules:

- `main_test.py`: This module is responsible for launching the simulation. It utilizes other modules, including:
  (i) Modules for handling training/validation data (dataloader generation, balancing, augmentation).
  (ii) Modules for training and validation.
  (iii) Modules for conducting ablation studies on the trained model.

- `data_cleaning.py`: This module is responsible for automatic data cleaning. Specifically, during data analysis, it was observed that one class contained two images without the corresponding binary mask, so those images were removed. It also provides a function to delete images generated by the "data_augmentation.py" module.

- `dataloader_utils.py`: This module is responsible for generating well-balanced dataloaders for training and validation.

- `data_augmentation.py`: This module provides automated support for the data augmentation process.

- `data_balancing.py`: This module is responsible for creating a training dataloader in which examples from all classes are presented an equal number of times, ensuring fair learning of the classes by the network.

- `metrics.py`: This module contains various metrics and loss functions used during training and validation.

- `model.py` and `arc_change_net.py`: These modules contain an implementation of the UNet network architecture. In particular, the second module offers a more customizable architecture in terms of depth (number of network layers), width (number of filters i.e., neurons per layer), use of normalization techniques, and more.

- `plotting_utils.py`: This module contains several useful functions for plotting metrics, results, kernels, activations, and more.

- `solver.py`: This module includes various methods for training, validation, and conducting ablation studies after training the model. It also provides functionality for saving and loading the model, visualizing model weights, activations, and kernels.

- `ablation_studies.py`: This module is responsible for executing ablation studies based on sensitivity analysis and pruning of network modules.

<!-- - results_analysis.py: This module (ad-hoc) is specifically created to obtain the best network configurations based on the analysis of statistics collected during training. Additionally, it allows for visualizing the results of ablation studies, based on the analysis of statistics collected during the execution of ablation studies.-->

The **results** are contained within the `final_considerations.pdf` file.

**The dataset can be downloaded at:** https://paperswithcode.com/dataset/ebhi-seg

**Other interesting resources:**

- **U-Net: Convolutional Networks for Biomedical Image Segmentation:** https://arxiv.org/abs/1505.04597
- **EBHI-Seg: A Novel Enteroscope Biopsy Histopathological Haematoxylin and Eosin Image Dataset for Image Segmentation Tasks:** https://arxiv.org/abs/2212.00532
- **Ablation Studies in Artificial Neural Networks:** https://arxiv.org/abs/1901.08644

## Some results:

| Model                   | Validation-loss | Macro Dice-Ratio | Macro Precision | Macro Recall | Macro F1-Score |
|-------------------------|-----------------|------------------|-----------------|--------------|----------------|
| PaperConfiguration      | -               | 0.8345           | 0.7595          | 0.7741       | 0.7667         |
| MyUNet18-0              | **0.0542**      | 0.9483           | 0.9621          | 0.9411       | 0.9514         |
| MyUNet3-0               |0.0566           | 0.9461           | 0.9412          | **0.9586**   | 0.9498         |
| MyUNetFinal             | 0.0564          | **0.9506**       | **0.9639**      | 0.9443       | **0.9539**     |
| MyUNet0-3 (worst)       | 0.2237          | 0.7642           | 1.000           | 0.6289       | 0.7721         |

## The parameters that can be provided through the command line and allow customization of the execution are:


| Argument              | Description                                                                  |
|-----------------------|------------------------------------------------------------------------------|
| run_name              | The name assigned to the current run                                         |
| model_name            | The name of the model to be saved or loaded                                  |
| epochs                | The total number of training epochs                                          |
| bs_train              | The batch size for training data                                             |
| bs_test               | The batch size for test data                                                 |
| workers               | The number of workers in the data loader                                     |
| print_every           | The frequency of printing losses during training                             |
| random_seed           | The random seed used to ensure reproducibility                               |
| lr                    | The learning rate for optimization                                           |
| loss                  | The loss function used for model optimization                                |
| val_custom_loss       | The weight of the jac_loss in the overall loss function                      |
| opt                   | The optimizer used for training                                              |
| early_stopping        | The threshold for early stopping during training                             |
| resume_train          | Determines whether to load the model from a checkpoint                       |
| use_batch_norm        | Indicates whether to use batch normalization layers in each conv layer       |
| use_double_batch_norm | Indicates whether to use 2 batch normalization layers in each conv layer     |
| use_inst_norm         | Indicates whether to use instance normalization layers in each conv layer    |
| use_double_inst_norm  | Indicates whether to use 2 instance normalization layers in each conv layer  |
| weights_init          | Determines whether to use weights initialization                             |
| th                    | The threshold used to split the dataset into train and test subsets          |
| dataset_path          | The path to save or retrieve the dataset                                     |
| checkpoint_path       | The path to save the trained model                                           | 
| pretrained_net        | Indicates whether to load a pretrained model on BRAIN-MRI                    |
| arc_change_net        | Indicates whether to load a customizable model                               |
| features              | A list of feature values (number of filters i.e., neurons in each layer)     |
| norm_input            | Indicates whether to normalize the input data                                |
| apply_transformations | Indicates whether to apply transformations to images and corresponding masks |
| dataset_aug           | Determines the type of data augmentation applied to each class               |
| balanced_trainset     | Generates a well-balanced train_loader for training                          |
| global_ablation       | Initiates an ablation study using global_unstructured or l1_unstructured     |
| grouped_pruning       | Initiates an ablation study using global_unstructured                        |
| all_one_by_one        | Initiates an ablation study using random_structured or random_unstructured   |
| selective_ablation    | Initiates an ablation study using random_structured or random_unstructured   |
| single_mod            | Initiates an ablation study with only one module                             |
| double_mod            | Initiates an ablation study with two modules                                 |
| random_structured     | Initiates an ablation study using random_structured                          |
| random_unstructured   | Initiates an ablation study using random_unstructured                        |
| conv2d_prune_amount   | The amount of pruning applied to conv2d layers                               |
| linear_prune_amount   | The amount of pruning applied to linear layers                               |
| num_iterations        | The number of iterations for the pruning process                             |
| weights_distr_histo   | Plots the histogram of weights distribution                                  |
| plt_weights_distr     | Plots the filters as images to visualize CNN kernels                         |
| no_skeep_7            | Removes the first skip connection from the bottom up                         |
| no_skeep_5            | Removes the second skip connection from the bottom up                        |
| no_skeep_3            | Removes the third skip connection from the bottom up                         |
| no_skeep_1            | Removes the last skip connection from the bottom up                          |

### Prerequisites

- [Python](https://www.python.org/downloads/) 3.5 or later installed on your system.
- The following modules:
  - [os](https://docs.python.org/3/library/os.html)
  - [json](https://docs.python.org/3/library/json.html)
  - [torch](https://pytorch.org/)
  - [numpy](https://numpy.org/)
  - [tqdm](https://tqdm.github.io/)
  - [matplotlib](https://matplotlib.org/)
  - [torchvision](https://pytorch.org/vision/stable/index.html)
  - [torch.nn.utils.prune](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
  - [cv2](https://docs.opencv.org/4.5.2/)
  - [PIL](https://pillow.readthedocs.io/en/stable/)
  - [albumentations](https://albumentations.ai/)
  - [torch.utils.data](https://pytorch.org/docs/stable/data.html)
  - [math](https://docs.python.org/3/library/math.html)
  - [random](https://docs.python.org/3/library/random.html)
  - [argparse](https://docs.python.org/3/library/argparse.html)
  - [torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html)
  - [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)
  - [pandas](https://pandas.pydata.org/)
  - [torch.optim](https://pytorch.org/docs/stable/optim.html)

Make sure to install these modules using `pip` or any other package manager like `miniconda` before running the code.

- [Python](https://www.python.org/downloads/) 3.5 or later installed on your system.
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system (optional but recommended).

### Installation

1. Clone this repository to your local machine or download the ZIP file and extract its contents.

   ```shell
   git clone https://github.com/your-username/repository-name.git
   ```

2. Navigate to the project directory.

   ```shell
   cd repository-name
   ```

3. (Optional) Create a virtual environment using Miniconda to isolate the project dependencies. If you don't have Miniconda installed, you can skip this step and proceed with the regular package installation.

   - Create a new virtual environment.

     ```shell
     conda create -n myenv python=3.9
     ```

   - Activate the virtual environment.

     - For Windows:

       ```shell
       conda activate myenv
       ```

     - For macOS/Linux:

       ```shell
       source activate myenv
       ```

4. Install the required modules.

   ```shell
   conda install -c conda-forge matplotlib pytorch numpy tqdm torchvision opencv pillow pandas
   pip install albumentations torchmetrics
   ```

5. Download the dataset from [https://figshare.com/articles/dataset/EBHI-SEG/21540159/1](https://figshare.com/articles/dataset/EBHI-SEG/21540159/1) and extract it to a suitable location on your system.

### Usage

Once you have completed the installation steps and downloaded the dataset, you need to define the correct file paths in the main_test.py file. Open the main_test.py file and modify the default value of the corresponding argparse parameter or provide the correct path through the command line to the appropriate paths to the dataset on your system.

After setting the correct file paths, you can run the `main_test.py` file.

```shell
python main_test.py --run_name run_final_2t --model_name unet_final_2t --random_seed 2 --opt Adam --arc_change_net --use_inst_norm --lr 0.001 --weights_init --balanced_trainset --early_stopping 6
```

Make sure you are in the project directory and have activated your virtual environment (if applicable) before running the above command.

### Additional Notes

- Modify the `main_test.py` file according to your needs or replace it with your own Python script.

Happy coding!


# Studi di ablazione dell'architettura di rete UNet addestrata sul dataset EBHI-Seg (ITA)

Il task di deep-learning in questione consiste nell'esecuzione di studi di ablazione sull'architettura di rete UNet addestrata sul dataset EBHI-Seg. Pertanto, i moduli Python che compongono il progetto sono:

- `main_test.py`: è il modulo responsabile del lancio della simulazione. Al suo interno, vengono utilizzati altri moduli, tra cui:
  (i) moduli per la gestione dei dati di training/validation (generazione dataloader, bilanciamento, aumento).
  (ii) moduli per il training e la validazione.
  (iii) moduli per eseguire gli studi di ablazione sul modello addestrato.

- `data_cleaning.py`: è il modulo responsabile per la pulizia automatica dei dati. In particolare, durante l'analisi dei dati, è stato osservato che una classe conteneva due immagini senza la corrispondente maschera binaria, quindi sono state rimosse. Inoltre, fornisce una funzione per eliminare le immagini generate dal modulo "data_augmentation.py".

- `dataloader_utils.py`: è il modulo responsabile per la generazione di dataloader ben bilanciati per il training e la validazione.

- `data_augmentation.py`: è il modulo respondabile per supporto automatico al processo di data-augmentation.

- `data_balancing.py`: è il modulo responsabile per la creazione di un dataloader di training in cui gli esempi di tutte le classi vengono presentati lo stesso numero di volte, garantendo così un apprendimento equo delle classi da parte della rete.

- `metrics.py`: è un modulo che contiene diverse metriche e funzioni di loss utilizzate durante il training e la validazione.

- `model.py` e `arc_change_net.py`: sono moduli che contengono un'implementazione dell'architettura di rete UNet. In particolare, il secondo modulo offre un'architettura più personalizzabile in termini di profondità (numero di layer della rete), ampiezza (numero di neuroni o filtri per ogni layer), uso di tecniche di normalizzazione e altro.

- `plotting_utils.py`: contiene alcune funzioni utili per il plotting di metriche, kernel, attivazioni e altro ancora.

- `solver.py`: è il modulo che include vari metodi per il training, la validazione e l'esecuzione degli studi di ablazione dopo l'addestramento del modello. Fornisce anche funzionalità per salvare e caricare il modello, visualizzare i pesi del modello, le attivazioni e i kernel.

- `ablation_studies.py`: è il modulo responsabile per l'esecuzione degli studi di ablazione basati sull'analisi di sensitività e il pruning dei moduli della rete.

<!-- - results_analysis.py: è un modulo (ad-hoc) appositamente creato per ottenere le migliori configurazioni di rete basandosi sull'analisi delle statistiche raccolte durante il training. Inoltre, consente di visualizzare i risultati degli studi di ablazione, basandosi sempre sull'analisi delle statistiche raccolte durante l'esecuzione degli studi di ablazione. -->
