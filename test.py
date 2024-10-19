from tensorflow.keras.preprocessing import image # type: ignore
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class ModelTester:
    def __init__(self, model_path, class_labels):

        self.model = tf.keras.models.load_model(model_path)
        self.class_labels = class_labels

    def preprocess_image(self, img_path, target_size=(224, 224)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, img_path):
        processed_image = self.preprocess_image(img_path)

        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions)
        class_label = self.class_labels[predicted_class]
        confidence = np.max(predictions) * 100

        print(f"The image belongs to the class '{class_label}' with a confidence of {confidence:.2f}%")

        plt.imshow(image.load_img(img_path))
        plt.title(f"Predicted: {class_label} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    model_path = 'modelo_custom.keras'
    class_labels = ['fruta_doscuartos', 'fruta_madura', 'fruta_sobremadura', 'fruta_trescuartos', 'fruta_verde']

    tester = ModelTester(model_path, class_labels)
    test_image_path = 'dataset/fruta_verde/1_V.jpg' 
    tester.predict(test_image_path)
