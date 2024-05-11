import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def calculate_percentage(hsv_image, yellow_range, brown_range):
    # Extract the hue channel
    hue_channel = hsv_image[:, :, 0]

    # Count pixels within the yellow and brown hue ranges
    yellow_pixels = np.sum(np.logical_and(hue_channel >= yellow_range[0], hue_channel <= yellow_range[1]))
    brown_pixels = np.sum(np.logical_and(hue_channel >= brown_range[0], hue_channel <= brown_range[1]))

    # Calculate the total number of pixels in the image
    total_pixels = hsv_image.shape[0] * hsv_image.shape[1]

    # Calculate the percentage of yellow and brown pixels
    yellow_percentage = (yellow_pixels / total_pixels) * 100
    brown_percentage = (brown_pixels / total_pixels) * 100
    
    return brown_percentage, yellow_percentage

def detect_shadow(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold the Value channel to detect shadows
    lower_shadow = np.array([0, 0, 0], dtype=np.uint8)
    upper_shadow = np.array([180, 255, 50], dtype=np.uint8)
    shadow_mask = cv2.inRange(hsv_image, lower_shadow, upper_shadow)

    # Count the number of shadow pixels
    num_shadow_pixels = np.count_nonzero(shadow_mask)

    # Calculate the percentage of shadowed area
    total_pixels = image.shape[0] * image.shape[1]
    percentage_shadow = (num_shadow_pixels / total_pixels) * 100

    return percentage_shadow

def process_images_in_folder(folder_path, output_folder_healthy, output_folder_unhealthy, max_brown_percentage , max_yellow_percentage, max_shadow_percentage):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: unable to load image '{image_file}'")
            continue
        # Detect shadows
        shadow_percentage = detect_shadow(image)

        # Convert BGR image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define the ranges of green, yellow, and brown hues in the HSV color space
        green_range = [30, 90]   # Define the hue range for green
        yellow_range = [15, 30]  # Define the hue range for yellow
        brown_range = [5, 15]    # Define the hue range for brown
        
        # Calculate percentages for this image
        brown_percentage, yellow_percentage = calculate_percentage(hsv_image, yellow_range, brown_range)
        
        # Determine the folder to save the image based on brown and yellow percentages
        if brown_percentage > max_brown_percentage or yellow_percentage > max_yellow_percentage or shadow_percentage > max_shadow_percentage:
            output_folder = output_folder_unhealthy
        else:
            output_folder = output_folder_healthy
        
        # Save the image to the appropriate folder
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)

        print(f"Processed and saved image '{image_file}' in folder '{output_folder}'.")


# Define the main folder containing subfolders with images
main_folder_path = 'C:\\Users\\15123\\Desktop\\groupProjectIslam'

# Define the subfolder names
subfolders = ['curl_stage1', 'curl_stage2', 'curl_stage1+sooty', 'curl_stage2+sooty', 'healthy']

# Create output folders for healthy and unhealthy images
output_folder_healthy = os.path.join(main_folder_path, 'healthiswealth')
output_folder_unhealthy = os.path.join(main_folder_path, 'unhealthyy')
output_folder_results = os.path.join(main_folder_path, 'results')

os.makedirs(output_folder_healthy, exist_ok=True)
os.makedirs(output_folder_unhealthy, exist_ok=True)
os.makedirs(output_folder_results, exist_ok=True)

# Calculate maximum percentages only from the images in the healthy folder
max_shadow_percentage = 0
max_brown_percentage = 0
max_yellow_percentage = 0

healthy_folder_path = os.path.join(main_folder_path, 'healthy')
healthy_image_files = [f for f in os.listdir(healthy_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in healthy_image_files:
    image_path = os.path.join(healthy_folder_path, image_file)
    image = cv2.imread(image_path)
    
    # Detect shadows
    shadow_percentage = detect_shadow(image)
    
    # Convert BGR image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the ranges of green, yellow, and brown hues in the HSV color space
    green_range = [30, 90]   # Define the hue range for green
    yellow_range = [15, 30]  # Define the hue range for yellow
    brown_range = [5, 15]    # Define the hue range for brown
    
    # Calculate percentages for this image
    brown_percentage, yellow_percentage = calculate_percentage(hsv_image, yellow_range, brown_range)
    
    # Update maximum percentages if necessary
    if brown_percentage > max_brown_percentage:
        max_brown_percentage = brown_percentage
        
    if yellow_percentage > max_yellow_percentage:
        max_yellow_percentage = yellow_percentage
        
    if shadow_percentage > max_shadow_percentage:
        max_shadow_percentage = shadow_percentage

# Print out maximum values
print(f"Max Brown Percentage: {max_brown_percentage}")
print(f"Max Yellow Percentage: {max_yellow_percentage}")
print(f"Max Shadow Percentage: {max_shadow_percentage}")

# Process images in each subfolder
for subfolder in subfolders:
    folder_path = os.path.join(main_folder_path, subfolder)
    process_images_in_folder(folder_path, output_folder_healthy, output_folder_unhealthy, max_brown_percentage, max_yellow_percentage, max_shadow_percentage)

# Path to the folder containing the predicted labels from your code
predicted_folder = "C:\\Users\\15123\\Desktop\\groupProjectIslam\\healthiswealth"

# Path to the healthy folder
healthy_folder = "C:\\Users\\15123\\Desktop\\groupProjectIslam\\healthy"

# Get the list of image files in the predicted and healthy folders
predicted_files = os.listdir(predicted_folder)
healthy_files = os.listdir(healthy_folder)

# Initialize counters
total_images = len(predicted_files)
incorrect_predictions = 0

# Iterate over each image file in the predicted folder
for filename in predicted_files:
    # Check if the image file is not in the healthy folder
    if filename not in healthy_files:
        incorrect_predictions += 1
for subfolder in subfolders:
    folder_path = os.path.join(main_folder_path, subfolder)
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    total_images += len(image_files)

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        image_color, image_gray = load_image(image_path)
        hist_eq_image = histogram_equalization(image_gray)
        erosion, dilation, opening, closing = morphological_operations(image_gray)
        contrast_image = contrast_stretching(image_gray)
        
       
        # Display the results
        plt.figure(figsize=(15, 10))
        plt.subplot(338), plt.imshow(image, cmap='gray'), plt.title('color')
        plt.subplot(331), plt.imshow(image_gray, cmap='gray'), plt.title('Original')
        plt.subplot(332), plt.imshow(hist_eq_image, cmap='gray'), plt.title('Histogram Equalization')
        plt.subplot(333), plt.imshow(contrast_image, cmap='gray'), plt.title('Contrast Stretching')
        plt.subplot(334), plt.imshow(erosion, cmap='gray'), plt.title('Erosion')
        plt.subplot(335), plt.imshow(dilation, cmap='gray'), plt.title('Dilation')
        plt.subplot(336), plt.imshow(opening, cmap='gray'), plt.title('Opening')
        plt.subplot(337), plt.imshow(image_gray, cmap='gray'), plt.title('Detected Shapes')
        
        plot_path = os.path.join(output_folder_results, f"{os.path.splitext(image_file)[0]}_plot.png")
        plt.savefig(plot_path)
        plt.close()
        
        if image is None:
            print(f"Error: unable to load image '{image_file}'")
            continue
# Calculate accuracy
accuracy = ((total_images - incorrect_predictions) / total_images) * 100
print(f"Accuracy: {accuracy:.2f}%")

print("Images sorted and saved into folders.")
