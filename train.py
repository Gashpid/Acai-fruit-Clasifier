from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import tensorflow as tf # type: ignore
import matplotlib.pyplot as plt
from modules.sysprop import *
import seaborn as sns # type: ignore
import numpy as np
import yaml

def loadConfig(config_file='scripts/venv/repo/config.yaml', label='ProjectPath'):
    if not os.path.exists(config_file):
        print(f"El archivo {config_file} no existe.")
        return None
    try:
        with open(config_file, 'r') as file:
            config_data = yaml.safe_load(file)

        if label in config_data:
            project_path = config_data[label]
            return project_path
        else:
            return None
    except yaml.YAMLError as e:
        return None
    
class Trainer(object):
    epochs = 2
    batch_size= 16
    optimizers = {'rmsprop': tf.keras.optimizers.RMSprop(),
                  'adam': tf.keras.optimizers.Adam(), 
                  'sgd': tf.keras.optimizers.SGD()} 
    
    def __init__(self, project_path):
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-4)
        self.model_path = os.path.join(project_path, "model.keras")
        self.dataset = os.path.join(project_path, "dataset_mod")

        print ("pat_1",self.model_path)
        print("pat_2", self.dataset)
        self.optimizer = self.optimizers['adam']
        self.model = Sequential()
        self.layers()
    
    def layers(self):
        # Convolution and pooling input layer
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second convolution and pooling layer
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Third convolution and pooling layer
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flattening and dense layers
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
        self.model.add(Dropout(0.6))
        self.model.add(Dense(5, activation='softmax'))  # 5 classes

        self.model.summary()

    def __weight_calc__(self, path):
        folder_counts = {}

        print('Path in', path)

        for index, folder_name in enumerate(os.listdir(path)):
            folder_path = os.path.join(path, folder_name)

            if os.path.isdir(folder_path):
                file_count = len(os.listdir(folder_path))
                folder_counts[index] = file_count
                print(file_count)
        
        max_value = max(folder_counts.values())

        for key in folder_counts:
            if(folder_counts[key] == max_value):
                folder_counts[key] = folder_counts[key]/max_value
            else:
                folder_counts[key] = 1+(folder_counts[key]/max_value)
        
        self.weights = folder_counts    
    
    def __data_prepare__(self):
        self.__weight_calc__(self.dataset)
        self.datagen = ImageDataGenerator(
            brightness_range=[0.8, 1.2],
            height_shift_range=0.2,
            width_shift_range=0.2,
            validation_split=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            vertical_flip=True,
            rotation_range=30,
            rescale=1.0/255,
            shear_range=0.2,
            zoom_range=0.2
        )
    
    def __model_prepare__(self):
        self.train_generator = self.datagen.flow_from_directory(
            self.dataset,
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        self.validation_generator = self.datagen.flow_from_directory(
            self.dataset,
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
    
    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.__data_prepare__()
        self.__model_prepare__()
    
    def train(self):
        self.history = self.model.fit(
            self.train_generator,
            validation_steps=self.validation_generator.samples // self.batch_size,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            callbacks=[self.early_stopping, self.reduce_lr],
            validation_data=self.validation_generator,
            class_weight=self.weights,
            epochs=self.epochs
        )
        
        self.model.save(self.model_path)
    
    def model_eval(self):
        _, val_acc = self.model.evaluate(self.validation_generator)
        print(f"Accuracy in validation: {val_acc * 100:.2f}%")
    
    def model_report(self):
        predictions = self.model.predict(self.validation_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.validation_generator.classes
        self.class_labels = list(self.validation_generator.class_indices.keys())

        print(classification_report(true_classes, predicted_classes, target_names=self.class_labels))
    
    def confusion_matrix(self):
        self.Y_pred = self.model.predict(self.validation_generator)
        y_pred = np.argmax(self.Y_pred, axis=1)
        y_true = self.validation_generator.classes

        cm = confusion_matrix(y_true, y_pred)
        class_names = list(self.validation_generator.class_indices.keys())

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion matrix')
        plt.ylabel('True class')
        plt.xlabel('Class Predicted')
        plt.show()

    def roc_curve(self):
        fpr,roc_auc,tpr = {},{},{}

        for i in range(len(self.class_labels)):
            fpr[i], tpr[i], _ = roc_curve(self.validation_generator.labels == i, self.Y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        for i in range(len(self.class_labels)):
            plt.plot(fpr[i], tpr[i], label=f'Class {self.class_labels[i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (False Positive Rate)')
        plt.ylabel('True Positive Rate (TRP)')
        plt.title('ROC curves by class')
        plt.legend(loc='lower right')
        plt.show()
    

if __name__ == '__main__':
    path = loadConfig()

    if (path != None):
        model = Trainer(path)
        model.compile()
        model.train()

        model.model_eval()
        model.model_report()

        model.confusion_matrix()
        model.roc_curve()

        print("Training completed")