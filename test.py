from tensorflow.keras.preprocessing import image # type: ignore
from modules.plotter import HTMLPlotter
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import yaml
import os

def loadConfig(config_file='scripts/venv/repo/config.yaml', label1='ProjectPath', label2 ="ModelPath", label3="CurrentImage"):
    if not os.path.exists(config_file):
        return None,None,None,None
    try:
        with open(config_file, 'r') as file:
            config_data = yaml.safe_load(file)

        if label1 in config_data and label2 in config_data and label3 in config_data:
            current_image = config_data[label3]
            project_path = config_data[label1]
            model_path = config_data[label2]
            try:
                winMode = config_data["WindowMode"]
            except: winMode = None
            return project_path,model_path,current_image,winMode
        else:
            return None,None,None,None
    except yaml.YAMLError as e:
        return None,None,None,None
    
class ModelTester:
    def __init__(self, model_path, class_labels):

        self.model = tf.keras.models.load_model(model_path)
        self.class_labels = class_labels

    def preprocess_image(self, img_path, target_size=(224, 224)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, img_path, output_path, winMode):
        processed_image = self.preprocess_image(img_path)

        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions)
        class_label = self.class_labels[predicted_class]
        confidence = np.max(predictions) * 100

        print(f"The image belongs to the class '{class_label}' with a confidence of {confidence:.2f}%")

        fig = plt.figure(figsize=(12, 6))
        plt.imshow(image.load_img(img_path))
        plt.title(f"Predicted: {class_label} ({confidence:.2f}%)")
        plt.axis('off')

        file_name = os.path.join(output_path, "output\predicted.html")
        full_path = os.path.dirname(file_name)
        os.makedirs(full_path, exist_ok=True)
        HTMLPlotter(fig, file_name)
        if winMode: plt.close()
        else: plt.show()

if __name__ == '__main__':
    path,model,image_path,winMode = loadConfig()

    if path and model and image_path != None:
        class_labels = ['fruta_doscuartos', 'fruta_madura', 'fruta_sobremadura', 'fruta_trescuartos', 'fruta_verde']

        tester = ModelTester(model, class_labels)
        tester.predict(image_path, path, winMode)
