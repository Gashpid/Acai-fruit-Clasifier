import matplotlib.pyplot as plt
from bgremove import BGRemove
import cv2, os

class Crop():
    def __init__(self, show_diference=False, verbose=False):
        self.enable_plotter = show_diference
        self.bgremove = BGRemove()
        self.verbose = verbose

    def __load_image__(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from path: {image_path}")
        return image

    def __check_dimensions__(self, image):
        height, width, _ = image.shape
        return height, width

    def __crop_square__(self, image):
        height, width = image.shape[:2]
    
        # Determine the smaller dimension
        min_dimension = min(height, width)
        
        # Calculate offset for cropping from the center
        if height > width:
            offset = (height - min_dimension) // 2
            cropped_image = image[offset:offset + min_dimension, :]
        else:
            offset = (width - min_dimension) // 2
            cropped_image = image[:, offset:offset + min_dimension]

        return cropped_image
    
    def __save_image__(self, image, output_path):
        output_dir = os.path.dirname(output_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(output_path, image)

        if self.verbose:
            print(f"Cropped image saved to: {output_path}")

    def __display_images__(self, original, cropped):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        plt.title("Cropped Image")
        plt.axis('off')

        plt.show()

    def square(self, image_path, save_path, bgremove=False):
        original_image = self.__load_image__(image_path)
        height, width = self.__check_dimensions__(original_image)

        if bgremove:
            image_witout_bg = self.bgremove.remove(original_image)

        if height != width:
            if bgremove:
                cropped_image = self.__crop_square__(image_witout_bg)
            else:
                cropped_image = self.__crop_square__(original_image)

            self.__save_image__(cropped_image, save_path)

            if self.enable_plotter:
                self.__display_images__(original_image, cropped_image)

            return cropped_image
        else:
            if self.verbose:
                print("The image is already square.")
            return original_image


"""
crop = Crop(verbose=True)
# Path to the image
image_path_input = r'../dataset/fruta_verde/10_V.jpg'  # Replace with your image path
image_path_output = r'../dataset_mod/fruta_verde/10_V.jpg'  # Replace with your image path
cropped_image = crop.square(image_path_input,save_path=image_path_output)
"""