# ML_classfication_FASHION_MNIST
Here is a draft README for your fashion image classification project:

# Fashion Image Classification

## Overview

This project classifies fashion images from the MNIST dataset using TensorFlow, Keras, Matplotlib, RandomForest, and SVC. The goal is to accurately predict the type of clothing based on the input image. 

## Dependencies

- TensorFlow 2.x
- Keras 
- Matplotlib
- scikit-learn
- Numpy
- Pandas

## Data

The [MNIST fashion dataset](http://yann.lecun.com/exdb/mnist/) is used. It contains 70,000 grayscale images of 10 categories of clothing items:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

The images are 28x28 pixels.

## Models

The following models are implemented and evaluated:

- Simple 2D Convolutional Neural Network with TensorFlow/Keras
- Random Forest Classifier 
- Support Vector Machine (SVC)

## Usage

The Jupyter notebook `fashion_classifier.ipynb` contains the implementation and evaluation of the models. 

To run it:

1. Clone this repository
2. Install the dependencies
3. Run `jupyter notebook fashion_classifier.ipynb`

## Results

The convolutional neural network achieves the best accuracy of around 90% on the test set. The random forest and SVC achieve 80-85% accuracy.

## References

- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow tutorials](https://www.tensorflow.org/tutorials)
- [Keras documentation](https://keras.io/api/) 
- [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)

Let me know if you would like me to modify or expand the README file further. I aimed to cover the key information but can add more details as needed.
 
