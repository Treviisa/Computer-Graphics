import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    # Load an image in grayscale
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def histogram_equalization(image):
    # Apply Histogram Equalization
    return cv2.equalizeHist(image)

def morphological_operations(image):
    # Define the kernel size for morphological operations
    kernel = np.ones((5,5), np.uint8)
    
    # Erosion
    erosion = cv2.erode(image, kernel, iterations=1)
    
    # Dilation
    dilation = cv2.dilate(image, kernel, iterations=1)
    
    # Opening
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Closing
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    return erosion, dilation, opening, closing

def contrast_stretching(image):
    # Apply Contrast Stretching
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    
    # Stretching the histogram to the range [0, 255]
    stretched_image = 255 * (image - min_intensity) / (max_intensity - min_intensity)
    return stretched_image.astype(np.uint8)

# Example usage
image_path = 'path_to_your_image.jpg'
image = load_image(image_path)
hist_eq_image = histogram_equalization(image)
erosion, dilation, opening, closing = morphological_operations(image)
contrast_image = contrast_stretching(image)

# Display the results
plt.figure(figsize=(10, 7))
plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(232), plt.imshow(hist_eq_image, cmap='gray'), plt.title('Histogram Equalization')
plt.subplot(233), plt.imshow(contrast_image, cmap='gray'), plt.title('Contrast Stretching')
plt.subplot(234), plt.imshow(erosion, cmap='gray'), plt.title('Erosion')
plt.subplot(235), plt.imshow(dilation, cmap='gray'), plt.title('Dilation')
plt.subplot(236), plt.imshow(opening, cmap='gray'), plt.title('Opening')
plt.show()
