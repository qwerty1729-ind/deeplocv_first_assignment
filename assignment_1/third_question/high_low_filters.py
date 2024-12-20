import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to pad and resize images
def pad_resize(image, target_size):
    h, w = image.shape[:2]
    top = (target_size[0] - h) // 2
    bottom = target_size[0] - h - top
    left = (target_size[1] - w) // 2
    right = target_size[1] - w - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

# Function for high-pass filter
def high_pass_filter(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Function for low-pass filter
def low_pass_filter(image, kernel_size=(21, 21)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Main function
def main():
    # Load images
    image1 = cv2.imread("image1.jpg")
    image2 = cv2.imread("image2.jpg")

    if image1 is None or image2 is None:
        print("Error: Images not found!")
        return

    # Convert to RGB for Matplotlib
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Resize images to the same size using padding
    target_size = (max(image1.shape[0], image2.shape[0]), max(image1.shape[1], image2.shape[1]))
    image1_resized = pad_resize(image1, target_size)
    image2_resized = pad_resize(image2, target_size)

    # Apply filters
    high_pass_image = high_pass_filter(image1_resized)
    low_pass_image = low_pass_filter(image2_resized)

    # Combine high-pass and low-pass images
    combined_image = cv2.addWeighted(high_pass_image, 0.5, low_pass_image, 0.5, 0)

    # Set up GridSpec layout
    fig = plt.figure(figsize=(10, 8))
    grid = plt.GridSpec(2, 3, width_ratios=[1, 1, 1])

    # 2x2 Grid (Left side)
    ax1 = fig.add_subplot(grid[0, 0])  # Top-left: Original Image 1
    ax1.imshow(image1_resized)
    ax1.set_title("Image 1")
    ax1.axis("off")

    ax2 = fig.add_subplot(grid[0, 1])  # Top-right: High-Pass Filter
    ax2.imshow(high_pass_image)
    ax2.set_title("High-Pass Filter")
    ax2.axis("off")

    ax3 = fig.add_subplot(grid[1, 0])  # Bottom-left: Original Image 2
    ax3.imshow(image2_resized)
    ax3.set_title("Image 2")
    ax3.axis("off")

    ax4 = fig.add_subplot(grid[1, 1])  # Bottom-right: Low-Pass Filter
    ax4.imshow(low_pass_image)
    ax4.set_title("Low-Pass Filter")
    ax4.axis("off")

    # Middle Combined Image (Centrally aligned)
    ax5 = fig.add_subplot(grid[:, 2])  # Entire 3rd column
    ax5.imshow(combined_image)
    ax5.set_title("Combined Image")
    ax5.axis("off")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
