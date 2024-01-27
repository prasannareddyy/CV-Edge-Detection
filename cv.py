#K. Prasanna Reddy (1602-20-737-093)
#K. Praneeth(1602-20-737-092)
#Group no: 8
#IT-B

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image = cv2.imread('coin.jpeg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is None or image.size == 0:
    print("Error: Could not open the image or the image is empty.")
    exit()

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny edge detection
canny_edges = cv2.Canny(blurred_image, 50, 150)

# Apply Laplacian edge detection
laplacian_edges = cv2.Laplacian(blurred_image, cv2.CV_64F)

# Display the original image and edges
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(2, 2, 2), plt.imshow(blurred_image, cmap='gray'), plt.title('Gaussian Blurred Image')

plt.subplot(2, 2, 3), plt.imshow(canny_edges, cmap='gray'), plt.title('Canny Edges')
plt.subplot(2, 2, 4), plt.imshow(np.uint8(np.abs(laplacian_edges)), cmap='gray'), plt.title('Laplacian Edges')

plt.show()
