from PIL import Image
import numpy as np


def crop_flag_precisely(image_path):
    # Load the image
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure the image is in RGB mode

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Define thresholds for detecting red and white regions
    red = np.array([255, 0, 0])
    white = np.array([255, 255, 255])
    red_tolerance = 120
    white_tolerance = 150

    # Create a binary mask for the flag region
    mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            pixel = img_array[i, j]
            if np.linalg.norm(pixel - red) < red_tolerance:
                mask[i, j] = 1  # Mark red region
            elif np.linalg.norm(pixel - white) < white_tolerance and np.mean(pixel) > 200:
                mask[i, j] = 1  # Mark white region

    # Calculate row and column sums to find the flag boundaries
    row_sums = mask.sum(axis=1)
    col_sums = mask.sum(axis=0)

    # Detect rows and columns with significant pixel counts
    row_indices = np.where(row_sums > mask.shape[1] * 0.03)[0]  # Rows with >3% pixels
    col_indices = np.where(col_sums > mask.shape[0] * 0.03)[0]  # Columns with >3% pixels

    if row_indices.size > 0 and col_indices.size > 0:
        # Find the bounding box based on row/column indices
        top, bottom = row_indices[0], row_indices[-1]
        left, right = col_indices[0], col_indices[-1]

        # Apply minimal margin (or none)
        top = max(0, top)
        bottom = min(img_array.shape[0], bottom)
        left = max(0, left)
        right = min(img_array.shape[1], right)

        # Crop the flag region
        cropped_img = img.crop((left, top, right, bottom))
        return cropped_img
    else:
        raise ValueError("No flag detected in the image.")


def detect_flag_from_cropped(cropped_img):
    img = cropped_img.convert('RGB')

    # Resize the image for faster processing
    img = img.resize((100, 100))

    # Convert image to a numpy array
    img_array = np.array(img)

    # Define color thresholds for red and white
    red = np.array([255, 0, 0])
    white = np.array([255, 255, 255])

    # Segment the image into smaller regions
    region_size = 10
    red_count_top = 0
    white_count_top = 0
    red_count_bottom = 0
    white_count_bottom = 0

    for i in range(0, img_array.shape[0], region_size):
        for j in range(0, img_array.shape[1], region_size):
            region = img_array[i:i + region_size, j:j + region_size, :]
            avg_color = np.mean(region, axis=(0, 1))

            # Check the position of the region (top half or bottom half)
            if i < img_array.shape[0] // 2:
                if np.linalg.norm(avg_color - red) < 100:
                    red_count_top += 1
                elif np.linalg.norm(avg_color - white) < 100:
                    white_count_top += 1
            else:
                if np.linalg.norm(avg_color - red) < 100:
                    red_count_bottom += 1
                elif np.linalg.norm(avg_color - white) < 100:
                    white_count_bottom += 1

    # Determine the flag based on color counts
    if red_count_top > white_count_top and white_count_bottom > red_count_bottom:
        return "Indonesia"
    elif white_count_top > red_count_top and red_count_bottom > white_count_bottom:
        return "Poland"
    else:
        return "Unknown"


# Example usage
image_path = 'image.jpg'
cropped_flag = crop_flag_precisely(image_path)
# cropped_flag.show()  # Display the cropped flag
cropped_flag.save('cropped_flag_precisely.png')  # Save the cropped flag

result = detect_flag_from_cropped(cropped_flag)
print(f"The flag is: {result}")
