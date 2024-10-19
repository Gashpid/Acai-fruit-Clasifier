# Installation and Project Setup Guide

This document outlines the steps to install, prepare, and run the classification model in your project. Be sure to follow each step carefully to ensure everything works correctly.

## Prerequisites

- Python installed on your system (preferably version 3.6 or higher).
- Internet access to install necessary dependencies.

## 1. Virtual Environment Setup

The first step is to create and activate a virtual environment to isolate the project's dependencies and avoid conflicts.

### Steps to set up the virtual environment:
1. Create a virtual environment called `classifier_env`:

    ```bash
    python -m venv classifier_env
    ```

2. Activate the virtual environment:

    - On Windows:

    ```bash
    classifier_env\Scripts\activate.bat
    ```

    - On macOS/Linux:

    ```bash
    source classifier_env/bin/activate
    ```

3. Install the project dependencies using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

4. Disable OneDNN optimizations for TensorFlow (optional, but recommended if you experience performance issues):

    ```bash
    set TF_ENABLE_ONEDNN_OPTS=0
    ```

## 2. Preparing the Dataset

Before training the model, the dataset needs to be prepared. Make sure your images are organized and ready in the appropriate folders.

### Commands to prepare the dataset:
1. Navigate to the `modules` directory:

    ```bash
    cd modules
    ```

2. Run the script to prepare the dataset:

    ```bash
    python prepare_dataset.py
    ```

This script will process the images and organize them for model training.

## 3. Training the Model

Once the dataset is ready, you can proceed with training the model.

### Commands to train the model:
1. Return to the root project directory:

    ```bash
    cd ..
    ```

2. Run the training script:

    ```bash
    python train.py
    ```

This command will start the model training process. Make sure all training parameters are correctly configured in the `train.py` file.

## 4. Evaluating the Trained Model

After training the model, you may want to evaluate it with test data to check its performance.

### Command to test the model:
1. Run the following command:

    ```bash
    python test.py
    ```

This script will load the trained model and evaluate its accuracy using the test dataset.

---

## Additional Notes:
- Be sure to activate the virtual environment each time you work on the project by using the corresponding activation command for your operating system.
- If you encounter any dependency-related errors, verify that all dependencies are correctly installed by re-running `pip install -r requirements.txt`.
- If you have issues with TensorFlow and OneDNN, use the mentioned environment variable (`set TF_ENABLE_ONEDNN_OPTS=0`).


