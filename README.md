# Msc_Final_Project

This is my final project repository

dependencies: Keras, Tensorflow

This repository will be reworked in 3 days, where only necessary packages are kept, codes are commented, and packages or codes of the third party will be referenced.
# File Organization
## 1. File Structure
##### 1.1 this project folder contains 3 sub-folders, including A, B and Datasets
##### 1.2 A, B each contains code of corresponding task. There are three files, PreProcesssing.py, Model.py.
##### 1.3 Datasets should contain 6 folders, they are 4 folders including A1_dataset, A2_dataset, B1_dataset, B2_dataset and 2 other training dataset folder, celeb and cartoon-set.
##### 1.4 each Task_dataset folder contains logs file and txt file input
## 2.File Contents
##### 2.1 PreProcessing.py is used to pre-process the txt input
##### 2.2 Model.py is used to build up the model
##### 2.3 Main calls model and preprocessing functions to train the model
## 3. Package dependencies
##### 3.1 Python3
##### 3.2 pandas 1.0.3, tensorflow-gpu 2.1.0, and ekphrasis 0.5.1
## 4. Tensorboard usage
##### 4.1 Tensorboard is used to analyse and make best choice of model, it can plot the accuracy or loss against each epoch for all possible models.
##### 4.2 To use tensorboard, first install tensorbaord using anaconda.
##### 4.2 Open anaconda cmd window, type the following code
##### _tensorboard --logdir="log_file_path"_
##### 4.3 The logs file contains the plot of accuracy and loss of 27 models for each task.
##### 4.4 To choose the best model for each task, choose the one with the lowest and the most stable validation loss
##### 4.5 Once model is chosen, dense_layer, conv_layer, layer_size, epoch parameters are prepared for the main.py for accuracy test 
## 5. Comments
##### 5.1 since the structure of code for each task is similar, repetitive comments would only be made under Task A
