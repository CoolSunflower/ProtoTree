# ProtoTrees: Neural Prototype Trees for Interpretable Fine-grained Image Recognition

This repository contains the PyTorch code for Neural Prototype Trees (ProtoTrees), published at CVPR 2021: [&#34;Neural Prototype Trees for Interpretable Fine-grained Image Recognition&#34;](https://openaccess.thecvf.com/content/CVPR2021/html/Nauta_Neural_Prototype_Trees_for_Interpretable_Fine-Grained_Image_Recognition_CVPR_2021_paper.html).

The code is recreated as part of DA321M Course Project by Group Number 4:

- Adarsh Gupta (220101003)
- Arush Shaleen Mathur (220101017)
- Ayush Kumar (220101022)
- Tanay Goenka (220101098)
- Tanmay Mittal (220101099)
- Tanvi Doshi (220101102)

A ProtoTree is an intrinsically interpretable deep learning method for fine-grained image recognition. ProtoTree combines the representation power of a Convolutional Neural Network (CNN) with a structured decision tree, which enables human-understandable reasoning at each decision node. ProtoTree offers a novel approach to creating machine learning models that are accurate, interpretable, and closer to human reasoning, making it highly suitable for applications requiring trust and transparency.

## Features

* Interpretability: ProtoTree provides globally interpretable classification rules, displaying its reasoning process * through a binary decision tree.
* Fine-Grained Recognition: Achieves high performance in tasks with closely related classes, such as species of birds and types of cars.
* Prototype-Based Explanation: Each node contains a learned prototype, representing a visually relevant part of a class. Prototypes are shown as patches of training images.
* Efficient Decision Path: Pruned trees use minimal prototypes, enabling fast and interpretable decision paths.

## Requirements

* Python3
* PyTorch >= 1.5 and <= 1.7!
* Optional: CUDA
* numpy
* pandas
* opencv
* tqdm
* scipy
* matplotlib
* requests (to download the CARS dataset, or download it manually)
* gdown (to download the CUB dataset, or download it manually)

## Data

### CUB Dataset

1. from the main ProtoTree folder, run `python preprocess_data/cub.py` to create training and test sets

Download the following file: https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45 & Place it at ./data/CUB_200_2011/CUB-200-2011.tgz

Download [ResNet50 pretrained on iNaturalist2017](https://drive.google.com/drive/folders/1yHme1iFQy-Lz_11yZJPlNd9bO_YPKlEU) (Filename on Google Drive: `BBN.iNaturalist2017.res50.180epoch.best_model.pth`) and place it in the folder `features/state_dicts`.

### CARS Dataset

1. from the main ProtoTree folder, run `python preprocess_data/cars.py` to create training and test sets

Download Sources:

- https://www.kaggle.com/datasets/emanuelriquelmem/stanford-cars-pytorch:
  - Download cars_train and place at ./data/cars/cars_train
  - Download cars_test and place at ./data/cars/cars_test
  - Download car_devkit.tgz and place at ./data/cars/car_devkit.tgz
  - Extract car_devkit.tgz --> devkit/
  - Download cars_test_annos_withlabels.mat and place at ./data/cars/devkit/cars_test_annos_withlabels.mat
- https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset:
  - Download cars_annos.mat and place at ./data/cars/devkit/cars_annos.mat

## Training a ProtoTree

Create a folder ./runs

To train ProtoTree on Cars dataset:

python main_tree.py --epochs 100 --log_dir ./runs/prototree_cars --dataset CARS --lr 0.001 --lr_block 0.001 --lr_net 1e-5 --num_features 256 --depth 9 --net resnet50_inat --freeze_epochs 30 

To generate visualizations:

main_explain_local.py --log_fir ./runs/prototree_cars --dataset CARS --sample_dir ./data/cars/dataset/test/`<CLASSNAME>/<FILENAME> `
