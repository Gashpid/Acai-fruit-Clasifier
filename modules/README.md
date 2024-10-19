
# Utils Class Documentation

This `Utils` class provides several methods to process images, including background removal, color analysis, and geometric properties such as radius and circularity. This documentation explains each method in detail, making it accessible even to users without extensive programming knowledge.

## Dependencies
To use this class, the following libraries must be installed:
- `matplotlib`: for plotting graphs and images.
- `collections`: for counting occurrences of colors.
- `bgremove`: a library for removing the background from an image.
- `numpy`: for numerical operations.
- `opencv-python (cv2)`: for image processing.

You can install them with:
```bash
pip install matplotlib collections bgremove numpy opencv-python
```

## Initialization: `__init__`

```python
def __init__(self):
    self.__bgremove = BGRemove()
```
- Initializes the `Utils` class by creating an instance of the `BGRemove` object, which will later be used to remove the background from images.

---

## Method: `Tones`

```python
def Tones(self, image_path, exclude_threshold=100, plot=False, verbose=False):
```

This method analyzes the dominant colors in the image, removing the background first. It provides an option to plot the results or print the color values.

### Arguments:
- `image_path (str)`: The file path of the image to analyze.
- `exclude_threshold (int)`: A threshold to exclude overly bright colors. Default is 100.
- `plot (bool)`: If `True`, plots the image and the color tones. Default is `False`.
- `verbose (bool)`: If `True`, prints the dominant colors. Default is `False`.

### Returns:
- `color_array (np.ndarray)`: A NumPy array of the dominant colors in the image.

### How it Works:
1. Loads the image from the provided path.
2. Removes the background using the `BGRemove` class.
3. Finds the dominant colors, excluding those above the `exclude_threshold`.
4. If `plot` is enabled, it displays the original image and a color bar showing the predominant tones.

---

## Method: `colorIntensities`

```python
def colorIntensities(self, image_path, exclude_above=100, plot=False):
```

This method calculates the intensity of colors in an image by converting it to grayscale and plotting the intensity distribution.

### Arguments:
- `image_path (str)`: The file path of the image.
- `exclude_above (int)`: A threshold to exclude colors brighter than the specified value. Default is 100.
- `plot (bool)`: If `True`, plots the grayscale image and the intensity histogram.

### Returns:
- `intensities (np.ndarray)`: Array of color intensities.
- `bins (np.ndarray)`: Corresponding intensity values (0-255).

### How it Works:
1. Loads the image and removes its background.
2. Converts the image to grayscale.
3. Filters out intensity values above `exclude_above`.
4. If `plot` is enabled, it shows the original image and the histogram of intensity values.

---

## Method: `radius`

```python
def radius(self, img_path, plot=False):
```

This method calculates the radius of the largest circular object in the image by drawing the smallest enclosing circle around it.

### Arguments:
- `img_path (str)`: The file path of the image.
- `plot (bool)`: If `True`, plots the original and processed images showing the detected circle.

### Returns:
- `diameter (int)`: The diameter of the detected object in pixels.

### How it Works:
1. Loads the image and converts it to grayscale.
2. Applies a binary threshold to separate the object from the background.
3. Finds contours and determines the minimum enclosing circle.
4. If `plot` is enabled, displays the image with the detected circle drawn over it.

---

## Method: `circularity`

```python
def circularity(self, img_path, isCircle=0.6, plot=False):
```

This method computes the circularity of the largest object in the image, comparing its area to its perimeter. Circularity close to 1 indicates a perfect circle.

### Arguments:
- `img_path (str)`: The file path of the image.
- `isCircle (float)`: A threshold value to determine if the object is circular. Default is 0.6.
- `plot (bool)`: If `True`, plots the image and displays the circularity value.

### Returns:
- `circularity (float)`: The circularity value (1 for a perfect circle).
- `is_circle (bool)`: `True` if the object is classified as a circle based on the `isCircle` threshold.

### How it Works:
1. Loads the image and converts it to grayscale.
2. Applies a binary threshold to identify the object.
3. Calculates the area and perimeter of the largest contour.
4. Computes the circularity as `(4 * π * area) / (perimeter²)`.
5. If `plot` is enabled, displays the image with the contour and circularity information.

---

## Example Usage

Here’s how to use the class and its methods:

```python
utils = Utils()

# Display the tones of an image
utils.Tones('path_to_image.jpg', plot=True)

# Analyze color intensities in an image
utils.colorIntensities('path_to_image.jpg', plot=True)

# Calculate the radius of an object in the image
utils.radius('path_to_image.jpg', plot=True)

# Check the circularity of an object in the image
utils.circularity('path_to_image.jpg', plot=True)
```

---

## Notes
- **Background Removal**: This class uses `BGRemove` to automatically remove backgrounds from images before processing.
- **Plotting**: All methods have optional plotting functionality to help visualize the results.
- **Thresholding**: In methods like `Tones` and `colorIntensities`, thresholding is used to exclude extreme values and focus on relevant color data.
