import matplotlib.pyplot as plt
from collections import Counter
from plotter import HTMLPlotter
from bgremove import BGRemove
import numpy as np
import yaml
import cv2
import os

def loadConfig(config_file='scripts/venv/repo/config.yaml', label1='ProjectPath', label2 ="UtilsFucntion", label3="CurrentImage"):
    if not os.path.exists(config_file):
        return None,None,None,None
    try:
        with open(config_file, 'r') as file:
            config_data = yaml.safe_load(file)

        if label1 in config_data and label2 in config_data and label3 in config_data:
            current_image = config_data[label3]
            project_path = config_data[label1]
            utils_fucnt = config_data[label2]
            try:
                winMode = config_data["WindowMode"]
            except: winMode = None
            return project_path,utils_fucnt,current_image,winMode
        else:
            return None,None,None,None
    except yaml.YAMLError as e:
        return None,None,None,None

class Utils(object):
    def __init__(self):
        self.__bgremove = BGRemove()
        self.path = ""

    def __get_dominant_colors__(self, image, num_colors=5, exclude_threshold=None):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (100, 100))

        pixels = image_rgb.reshape(-1, 3)
        
        if exclude_threshold is not None:
            pixels = [pixel for pixel in pixels if all(value <= exclude_threshold for value in pixel)]
        
        color_counts = Counter(map(tuple, pixels))
        dominant_colors = color_counts.most_common(num_colors)
        dominant_colors.sort(key=lambda x: x[1])
        return dominant_colors

    def __create_color_array__(self, dominant_colors):
        color_array = np.array([color for color, _ in dominant_colors], dtype=np.uint8)
        return color_array

    def Tones(self, image_path, winMode, exclude_threshold=100, plot=False, verbose=False):
        print("Extracting tones of the image...")
        image = cv2.imread(image_path)

        image_wo_bg = self.__bgremove.remove(image)

        dominant_colors = self.__get_dominant_colors__(image_wo_bg, num_colors=5, exclude_threshold=exclude_threshold)

        color_array = self.__create_color_array__(dominant_colors)
        if verbose: print("Predominant colors:", color_array)

        if plot:
            heatmap = np.zeros((len(color_array), 10, 3), dtype=np.uint8)
            for i, color in enumerate(color_array):
                heatmap[i, :, :] = color

            dpi = 72
            fig_height = 6
            fig_width = 12
            max_subplot_width = 5 / dpi

            fig, axs = plt.subplots(1, 2, figsize=(fig_width, fig_height), gridspec_kw={'width_ratios': [1, max_subplot_width]})
            plt.subplots_adjust(wspace=0)

            axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axs[0].set_title('Original Image & Tones')
            axs[0].axis('off')

            color_subplot = np.zeros((len(color_array), 10, 3), dtype=np.uint8)
            for i, color in enumerate(color_array):
                color_subplot[i, :, :] = color

            axs[1].imshow(color_subplot, aspect='auto')
            axs[1].set_title('Dominant Colors')
            axs[1].set_xticks([])
            axs[1].set_yticks([])

            for i in range(len(color_array)):
                axs[1].text(5, i, f'Color {i + 1}', ha='center', va='center', fontsize=12, color='w')

            plt.tight_layout()
            file_name = os.path.join(self.path, "output\\tones.html")
            full_path = os.path.dirname(file_name)
            os.makedirs(full_path, exist_ok=True)
            HTMLPlotter(fig, file_name)

            print("Finished successful...")

            if winMode: plt.close()
            else: plt.show()

        return color_array

    def colorIntensities(self, image_path, winMode, exclude_above=100, plot=False):

        print("Computing color intensities of the image...")

        image = cv2.imread(image_path)

        image_wo_bg = self.__bgremove.remove(image)

        gray_image = cv2.cvtColor(image_wo_bg, cv2.COLOR_BGR2GRAY)

        if exclude_above is not None:
            mask = gray_image <= exclude_above
            filtered_image = gray_image[mask]
        else:
            filtered_image = gray_image

        intensities, bins = np.histogram(filtered_image.flatten(), bins=256, range=[0, 256])

        if plot:
            fig = plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.bar(bins[:-1], intensities, color='gray', width=1)
            plt.title(f'Intensities of Fruit Colour')
            plt.xlabel('Color Intensity')
            plt.ylabel('Frequency')

            plt.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

            plt.tight_layout()
            file_name = os.path.join(self.path, "output\intensities.html")
            full_path = os.path.dirname(file_name)
            os.makedirs(full_path, exist_ok=True)
            HTMLPlotter(fig, file_name)

            print("Finished successful...")

            if winMode: plt.close()
            else: plt.show()
        
        return intensities, bins
    
    def diameter(self, img_path, winMode, plot=False):
        print("Computing diameter of the fruit...")
        image = cv2.imread(img_path)

        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binarizada = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY_INV)
        contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        imagen_con_circulo = image.copy()
        diameter = 0

        if len(contornos) > 0:
            contorno_max = max(contornos, key=cv2.contourArea)
            (x, y), radio = cv2.minEnclosingCircle(contorno_max)

            centro = (int(x), int(y))
            radio = int(radio)
            diameter = radio*2

            cv2.circle(imagen_con_circulo, centro, radio, (0, 255, 0), 2)

        if plot:
            fig, axes = plt.subplots(1, 4, figsize=(16, 8))

            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original image")
            axes[0].axis('off')

            axes[1].imshow(gris, cmap='gray')
            axes[1].set_title("grayscale")
            axes[1].axis('off')

            axes[2].imshow(binarizada, cmap='gray')
            axes[2].set_title("Binarized image")
            axes[2].axis('off')

            axes[3].imshow(cv2.cvtColor(imagen_con_circulo, cv2.COLOR_BGR2RGB))
            axes[3].set_title("Wrap-around")
            axes[3].axis('off')
            axes[3].text(0, 1, 'Diameter [px]: '+str(diameter), fontsize=12, color='k', ha='left', va='top', transform=axes[3].transAxes)

            plt.tight_layout()
            file_name = os.path.join(self.path, "output\diameter.html")
            full_path = os.path.dirname(file_name)
            os.makedirs(full_path, exist_ok=True)
            HTMLPlotter(fig, file_name)

            print("Finished successful...")

            if winMode: plt.close()
            else: plt.show()
        
        return diameter

    def circularity(self, img_path, winMode, isCircle=0.6, plot=False):

        print("Computing circularity of the fruit")
        
        image = cv2.imread(img_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binarized = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_with_circle = image.copy()
        
        circularity = 0

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)

            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)

            cv2.drawContours(image_with_circle, [largest_contour], -1, (255, 0, 0), 5)

            if circularity < isCircle:
                is_circle = False
            else: 
                is_circle = True

        if plot:
            fig, axes = plt.subplots(1, 4, figsize=(16, 8))

            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            axes[1].imshow(gray, cmap='gray')
            axes[1].set_title("Grayscale")
            axes[1].axis('off')

            axes[2].imshow(binarized, cmap='gray')
            axes[2].set_title("Binarized Image")
            axes[2].axis('off')

            axes[3].text(0, 1, f'Circularity: {circularity:.2f}\n' + ('Is a Circle' if is_circle else 'Not a Circle'), fontsize=12, color='k', ha='left', va='top', transform=axes[3].transAxes)
            axes[3].imshow(cv2.cvtColor(image_with_circle, cv2.COLOR_BGR2RGB))
            axes[3].set_title("Enclosing Contour")
            axes[3].axis('off')
                
            plt.tight_layout()
            file_name = os.path.join(self.path, "output\circularity.html")
            full_path = os.path.dirname(file_name)
            os.makedirs(full_path, exist_ok=True)
            HTMLPlotter(fig, file_name)

            print("Finished successful...")

            if winMode: plt.close()
            else: plt.show()
        
        return circularity, is_circle


if __name__ == '__main__':
    path,mode,image,winMode = loadConfig()
    if path and mode != None:
        utils = Utils()
        utils.path = path
        if mode == "Tone":
            utils.Tones(image, winMode, plot=True)
        elif mode == "Intensities":
            utils.colorIntensities(image, winMode, plot=True)
        elif mode == "Diameter":
            utils.diameter(image, winMode, plot=True)
        elif mode == "Circularity":
            utils.circularity(image, winMode, plot=True)