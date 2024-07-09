import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread('palestineReza.jpg')  # Replace 'image_path.jpg' with the path to your image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.title("Original Image")
plt.show()

# Reshape the image to a 2D array of pixels
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Define criteria, number of clusters(K) and apply KMeans
k = 7  # Change this value for different number of clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert back to 8 bit values
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]

# Reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)
plt.imshow(segmented_image)
plt.title("Segmented Image")
plt.show()

# Display the clusters
labels = labels.flatten()
segmented_images = []
for i in range(k):
    masked_image = np.copy(image)
    masked_image = masked_image.reshape((-1, 3))
    masked_image[labels != i] = [0, 0, 0]
    masked_image = masked_image.reshape(image.shape)
    segmented_images.append(masked_image)
    plt.imshow(masked_image)
    plt.title(f"Cluster {i+1}")
    plt.show()