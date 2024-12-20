import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function for grey scaling
def grey_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function for thresholding to 2 levels (black and white)
def threshold_bw(image, threshold_value=128):
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

# Function for thresholding to 16 grey levels
def threshold_16_levels(image):
    factor = 256 // 16  # Divide the range of 0-255 into 16 intervals
    quantized = (image // factor) * factor
    return quantized

# Function to apply Sobel filter and Canny edge detection
def edge_detection(image):
    sobel = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)
    canny = cv2.Canny(image, 50, 150)
    return sobel, canny

# Function to apply Gaussian filter for noise removal (using a kernel)
def gaussian_blur(image, kernel_size=(5, 5)):
    kernel = np.array([[1, 4, 6, 4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1, 4, 6, 4, 1]]) / 256  # Normalize the kernel
    return cv2.filter2D(image, -1, kernel)

# Function to sharpen an image
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])  # Sharpening kernel
    return cv2.filter2D(image, -1, kernel)

# Function to capture and process image
def process_image():
    cap = cv2.VideoCapture(0)  # Open camera (use 1 for external camera if needed)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Capturing image...")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Cannot capture image.")
        return

    print("Processing image...")

    # Processing the captured image
    gray = grey_scale(frame)
    binary = threshold_bw(gray)
    grey_16 = threshold_16_levels(gray)
    sobel, canny = edge_detection(gray)
    blurred = gaussian_blur(gray)
    sharpened = sharpen_image(blurred)

    # Plot all images in a 2x4 grid using Matplotlib
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()  # Flatten the axes array

    titles = ['Gray Scale', 'Binary Threshold', '16 Grey Levels',
              'Sobel Filter', 'Canny Edge', 'Gaussian Blur', 'Sharpened Image', 'RGB to BGR']
    images = [gray, binary, grey_16, sobel, canny, blurred, sharpened, cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR)]

    for i in range(8):
        axes[i].imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_image()
