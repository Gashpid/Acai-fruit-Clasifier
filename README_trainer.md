# Documentation of Training a CNN Model for Image Classification

This project implements a convolutional neural network (CNN) model using TensorFlow and Keras, in order to classify images into five different classes. The functionality of the classes, functions, and variables included in the code is described in detail below.

## Table of Contents

- [Installation](#installation)
- [Code Description](#code-description)
- [Trainer Class](#trainer-class)
- [Important Functions](#important-functions)
- [Usage](#usage)
- [Results Display](#visualizing-results)

## Installation

Make sure you have the following packages installed before running the code:

```bash
pip install tensorflow keras matplotlib seaborn scikit-learn
```

## Code Description

### `Trainer` Class

The `Trainer` class is responsible for defining, compiling, training, and evaluating the CNN model.

#### Main Variables

- `epochs`: Number of epochs to train the model (default: 200).
- `batch_size`: Size of each batch of data (default: 16).
- `dataset`: Directory containing the images organized in folders according to their classes (default: 'dataset_mod').
- `optimizers`: Dictionary containing different optimizers (`adam`, `rmsprop`, `sgd`) that can be used to train the model.

#### Important Functions

- `__init__(self)`: Initializes the `Trainer` class, defines the callbacks for training (EarlyStopping and ReduceLROnPlateau) and builds the CNN model with the `layers` function.

- `layers(self)`: Defines the layers of the CNN model. The model has the following layers:
1. **Conv2D + MaxPooling2D**: For feature extraction from images.
2. **BatchNormalization**: To normalize the activation values ​​and speed up training.
3. **Flatten + Dense + Dropout**: To flatten the extracted features and feed them into a fully connected network with regularization.

- `__weight_calc__(self, path)`: Calculates class weights based on the number of images in each class. Weights are adjusted to deal with imbalances in the data.

- `__data_prepare__(self)`: Prepares the data generator for training and validation. Uses `ImageDataGenerator` to apply data augmentation (rotations, zooms, translations, etc.).

- `__model_prepare__(self)`: Loads the images from the directory, creating two data generators: one for training and one for validation.

- `compile(self)`: Compiles the model with the selected optimizer (`adam` by default) and the `categorical_crossentropy` loss function.

- `train(self)`: Trains the model using the data generators, applying callbacks to stop training when there is no improvement and adjust the learning rate.

- `test_model(self, img_path)`: Performs inference on a provided image, predicting its class.

- `model_eval(self)`: Evaluates the model's performance on the validation set, showing the accuracy.

- `model_report(self)`: Generates a detailed classification report, including metrics such as precision, recall, and f1-score for each class.

- `confusion_matrix(self)`: Calculates and displays a confusion matrix for the model's predictions on the validation set.

- `roc_curve(self)`: Calculates and draws ROC curves for each class, showing the model's performance in terms of true positive and false positive rates.

## Usage

Once the model has been trained, you can test it using the `test_model` method, passing the path of an image. Here is an example of how to train and evaluate the model:

```python
if __name__ == '__main__':
model = Trainer()
model.compile()
model.train()

model.model_eval()
model.model_report()

model.confusion_matrix()
model.roc_curve()

model.test_model('path/to/image.jpg')
```

## Visualizing Results

The code provides several ways to visualize the model results:
1. **Confusion Matrix**: Uses `seaborn` to draw a confusion matrix that helps to see how the model confuses classes.
2. **ROC Curves**: Displays ROC curves to evaluate the model's performance in binary or multiclass classification.