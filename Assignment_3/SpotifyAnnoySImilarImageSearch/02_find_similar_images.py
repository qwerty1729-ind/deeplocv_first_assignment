import os
import torch
from PIL import Image, ImageDraw
from torchvision import models, transforms
import torch.nn as nn
from annoy import AnnoyIndex

# Paths and parameters
images_folder = 'Cat'
output_folder = 'ImageDump'
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
images = os.listdir(images_folder)

# Load ResNet-18 with pretrained weights
weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
model.fc = nn.Identity()  # Remove the classification head

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure uniform size
    transforms.ToTensor(),
])

# Initialize and load Annoy Index
annoy_index = AnnoyIndex(512, "angular")
annoy_index.load('Cat_index.ann')

# Create a grid canvas for each query image
for i, image_name in enumerate(images):
    try:
        # Load and transform the query image
        image_path = os.path.join(images_folder, image_name)
        image = Image.open(image_path).convert('RGB')  # Ensure RGB mode
        input_tensor = transform(image).unsqueeze(0).to(device)

        if input_tensor.size(1) == 3:
            # Extract features and find nearest neighbors
            with torch.no_grad():
                output_tensor = model(input_tensor).cpu().numpy().squeeze()
            nns = annoy_index.get_nns_by_vector(output_tensor, 24)

            # Create a new grid image (1000x1000 for 5x5 grid)
            grid_image = Image.new('RGB', (1000, 1000))
            grid_image.paste(image.resize((200, 200)), (0, 0))  # Paste query image at top-left

            # Draw a red border around the query image
            draw = ImageDraw.Draw(grid_image)
            draw.rectangle([(0, 0), (199, 199)], outline='red', width=8)

            # Paste the nearest neighbors
            for j, nn_index in enumerate(nns):
                neighbor_image_path = os.path.join(images_folder, images[nn_index])
                neighbor_image = Image.open(neighbor_image_path).convert('RGB').resize((200, 200))
                x_offset = 200 * ((j + 1) % 5)
                y_offset = 200 * ((j + 1) // 5)
                grid_image.paste(neighbor_image, (x_offset, y_offset))

            # Save the grid image
            grid_image.save(os.path.join(output_folder, f'image_{i}.png'))
            print(f'Saved image_{i}.png')
    except Exception as e:
        print(f"Error processing image {image_name}: {e}")
