import matplotlib.pyplot as plt
from collections import Counter
from bgremove import BGRemove
import numpy as np
import cv2

class Utils(object):
    def __init__(self):
        self.__bgremove = BGRemove()

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

    def Tones(self, image_path, exclude_threshold=100, plot=False, verbose=False):
        image = cv2.imread(image_path)

        image_wo_bg = self.__bgremove.remove(image)

        dominant_colors = self.__get_dominant_colors__(image_wo_bg, num_colors=5, exclude_threshold=exclude_threshold)

        color_array = self.__create_color_array__(dominant_colors)
        if verbose: print("Predominant colors:", color_array)

        if plot:
            heatmap = np.zeros((len(color_array), 10, 3), dtype=np.uint8)
            for i, color in enumerate(color_array):
                heatmap[i, :, :] = color

            plt.figure(figsize=(12, 6))

            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image & Tones')
            plt.axis('off')

            y_labels = [len(color_array)-(i+1) for i in range(len(color_array))]

            cbar_ax = plt.gca().inset_axes([1.05, 0, 0.05, 1])
            cbar_ax.set_yticks(np.arange(len(color_array)))
            cbar_ax.imshow(heatmap, aspect='auto')
            cbar_ax.set_yticklabels(y_labels)
            cbar_ax.set_xticks([])

            plt.tight_layout()
            plt.show()

        return color_array

    def colorIntensities(self, image_path, exclude_above=100, plot=False):
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
            plt.figure(figsize=(12, 6))

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
            plt.show()
        
        return intensities, bins
    
    def diameter(self, img_path, plot=False):
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
            _, axes = plt.subplots(1, 4, figsize=(16, 8))

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
            plt.show()
        
        return diameter

    def circularity(self, img_path, isCircle=0.6, plot=False):
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
            _, axes = plt.subplots(1, 4, figsize=(16, 8))

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
            plt.show()
        
        return circularity, is_circle


if __name__ == '__main__':
    utils = Utils()
    utils.Tones('../dataset/fruta_verde/1_V.jpg', plot=True)
    utils.colorIntensities('../dataset/fruta_verde/1_V.jpg', plot=True)
    utils.diameter('../dataset/fruta_verde/1_V.jpg',plot=True)
    utils.circularity('../dataset/fruta_verde/1_V.jpg', plot=True)