from crop import Crop
from tqdm import tqdm
import yaml
import os

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

class PrepareImages():
    def __init__(self):
        self.Crop = Crop()

    def load_dataset(self, path):
        classes = []

        # Check if the provided path is valid
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path does not exist: {path}")

        # Loop through each item in the specified directory
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)  # Get the full path of the entry
            if os.path.isdir(full_path):  # Check if it is a directory
                images = os.listdir(full_path)  # List elements inside the subfolder
                classes.append((entry, images))  # Append the subfolder name and its elements

        return classes
    
    def CropImages(self, pathI, pathO, Class, images,bgremove):
        try:
            for image in tqdm(images, desc="Cropping from "+Class):
                output_path = os.path.join(pathO,Class,image)
                input_path = os.path.join(pathI,Class,image)
                self.Crop.square(input_path, output_path,bgremove=bgremove)
        except Exception as e:
            print(e)
    
    def prepare(self, pathI, pathO, bgremove=False):
        try:
            classes = self.load_dataset(pathI)
            for Class, images in classes:
                self.CropImages(pathI,pathO,Class,images,bgremove)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    path = loadConfig()
    if (path != None):
        pathI = path+'/dataset'
        pathO = path+'/dataset_mod'

        Prep = PrepareImages()
        Prep.prepare(pathI,pathO,bgremove=True)
