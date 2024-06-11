# CIFAR10 Image Classification

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Installation and Setup](#installation-and-setup)
4. [Project Structure](#project-structure)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Development](#model-development)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [Challenges and Improvements](#challenges-and-improvements)
10. [Future Work](#future-work)
11. [Acknowledgements](#acknowledgements)

## Project Overview
This project aims to develop a deep learning model to classify images in the CIFAR-10 dataset, which consists of 60,000 32x32 colour images in 10 different classes. 

## Dataset Description
The CIFAR-10 dataset includes 60,000 images divided into 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images. The classes are:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## Installation and Setup
To set up the project environment, follow these steps:
1. Clone the repository: `git clone https://github.com/winsonchow/cifar10-image-classification.git`
2. Navigate to the project directory: `cd cifar10-image-classification`
3. Install the required libraries: `pip install -r requirements.txt`

## Project Structure
- `data/`: Contains the dataset files.
- `cifar10-image-classification.ipynb`: Jupyter notebook with data exploration, data preprocessing, model development and model training.
- `README.md`: Project documentation.

## Data Preprocessing

## Model Development
The model is a Convolutional Neural Network (CNN) built using PyTorch. The architecture includes:
- Convolutional layers with ReLU activations and MaxPooling.
- Fully connected (linear) layers.
- Drop out layers as a form of regularisation.
- The model is trained using the Cross-Entropy loss function and the Adam optimizer.

## Evaluation Metrics
The primary evaluation metrics for this project are:
- Loss: The Cross-Entropy loss is used to measure the performance of the model.
- Accuracy: The percentage of correctly classified images.

## Results
After training the final model for 20 epochs, the following results were achieved on the test set:
Average Loss: 0.5832
Accuracy: 80.46%

## Challenges and Improvements
Challenges faced during the project included:
- Underfitting

Potential improvements include:
- Experimenting with different and more complex architectures to better capture the patterns in the data.
- Fine-tuning hyperparameters.
- Increasing the dataset size by implementing more advanced data augmentation techniques.

## Future Work
Future work could involve:
- Training on larger datasets or using transfer learning from pre-trained models.
- Implementing more sophisticated models such as ResNet.
- Hyperparameter tuning using grid search, random search or Bayesian Optimisation.

## Acknowledgements
Special thanks to Professor Tarapong Sreenuch for his guidance and support.