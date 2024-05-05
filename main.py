import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern

def load_image(image_path):
    # Load an image in color and grayscale
    image_color = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    return image_color, image_gray

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
    stretched_image = 255 * (image - min_intensity) / (max_intensity - min_intensity)
    return stretched_image.astype(np.uint8)

def texture_features(image):
    # Texture features from GLCM
    glcm = greycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    return contrast, dissimilarity, homogeneity, energy, correlation

def shape_features(image):
    # Detect edges using Canny and find contours
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for cnt in contours:
        shape = {}
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        shape['perimeter'] = perimeter
        shape['area'] = area
        shapes.append(shape)
    return shapes

# Example usage
image_path = 'path_to_your_image.jpg'
image_color, image_gray = load_image(image_path)
hist_eq_image = histogram_equalization(image_gray)
erosion, dilation, opening, closing = morphological_operations(image_gray)
contrast_image = contrast_stretching(image_gray)
texture = texture_features(image_gray)
shapes = shape_features(image_gray)

# Display the results
plt.figure(figsize=(12, 8))
plt.subplot(331), plt.imshow(image_gray, cmap='gray'), plt.title('Original')
plt.subplot(332), plt.imshow(hist_eq_image, cmap='gray'), plt.title('Histogram Equalization')
plt.subplot(333), plt.imshow(contrast_image, cmap='gray'), plt.title('Contrast Stretching')
plt.subplot(334), plt.imshow(erosion, cmap='gray'), plt.title('Erosion')
plt.subplot(335), plt.imshow(dilation, cmap='gray'), plt.title('Dilation')
plt.subplot(336), plt.imshow(opening, cmap='gray'), plt.title('Opening')
plt.subplot(337), plt.imshow(image_gray, cmap='gray'), plt.title('Detected Shapes')
for shape in shapes:
    print(f"Shape Area: {shape['area']}, Perimeter: {shape['perimeter']}")

# Display texture features
print(f"Texture Features - Contrast: {texture[0]}, Dissimilarity: {texture[1]}, Homogeneity: {texture[2]}, Energy: {texture[3]}, Correlation: {texture[4]}")

plt.show()
