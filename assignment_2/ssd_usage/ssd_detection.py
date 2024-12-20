import torch
from torchvision import models, transforms
import cv2
import numpy as np

# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained SSD model and move it to the chosen device
weights = models.detection.SSD300_VGG16_Weights.DEFAULT
model = models.detection.ssd300_vgg16(weights=weights).to(device)
model.eval()  # Set the model to evaluation mode

# Define a preprocessing function with aspect ratio preservation
def preprocess(image):
    # Get the original size
    h, w = image.shape[:2]

    # Calculate the scaling factor to resize the shorter side to 300
    scale = 300.0 / min(h, w)
    new_h = min(int(h * scale), 300)
    new_w = min(int(w * scale), 300)

    # Resize the image while preserving the aspect ratio
    resized_image = cv2.resize(image, (new_w, new_h))

    # Create a black background and place the resized image in the center
    padded_image = np.zeros((300, 300, 3), dtype=np.uint8)
    y_offset = (300 - new_h) // 2
    x_offset = (300 - new_w) // 2

    # Place the resized image into the padded image
    padded_image[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1]] = resized_image

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(padded_image).unsqueeze(0).to(device)

# Define a function to draw bounding boxes with labels
def draw_boxes(image, boxes, scores, labels, threshold=0.3):
    # Load the COCO class names (for SSD model)
    class_names = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'none', 'stop sign', 'parking meter',
        'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'none', 'backpack', 'umbrella', 'none', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'none', 'wine glass', 'cup', 'fork', 'knife',
        'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
        'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'none', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            # Ensure the label index is within bounds
            if 0 < label < len(class_names):
                label_name = class_names[label]
            else:
                label_name = "unknown"

            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label_name}: {score:.2f}', (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Open the video file
cap = cv2.VideoCapture('sample_video.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor = preprocess(frame)

    # Perform object detection
    with torch.no_grad():
        detections = model(input_tensor)

    # Extract bounding boxes, scores, and labels
    boxes = detections[0]['boxes'].cpu().numpy().astype(np.float32)  # Convert to float32 for scaling
    scores = detections[0]['scores'].cpu().numpy()
    labels = detections[0]['labels'].cpu().numpy()

    # Scale boxes to original image size
    height, width = frame.shape[:2]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - 0) * (width / 300.0)  # Use adjusted scaling
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - 0) * (height / 300.0)

    # Convert back to int32 after scaling
    boxes = boxes.astype(np.int32)

    # Draw bounding boxes with labels
    output_frame = draw_boxes(frame, boxes, scores, labels)

    # Write the frame into the output video
    out.write(output_frame)

# Release everything when the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
