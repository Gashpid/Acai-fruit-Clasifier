from crop import Crop
from tqdm import tqdm
import os

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
    pathI = '../dataset'
    pathO = '../dataset_mod'

    Prep = PrepareImages()
    Prep.prepare(pathI,pathO,bgremove=True)

    

