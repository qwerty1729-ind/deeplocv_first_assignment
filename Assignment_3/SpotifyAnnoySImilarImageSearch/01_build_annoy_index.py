import os
import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from annoy import AnnoyIndex

# Path to your images folder
images_folder = 'Cat'
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

# Initialize Annoy Index
annoy_index = AnnoyIndex(512, "angular")

# Process images
for i, image_name in enumerate(images):
    image_path = os.path.join(images_folder, image_name)
    try:
        # Load and transform image
        image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Extract features using the model
        output_tensor = model(input_tensor).cpu().detach().numpy().squeeze()

        # Add features to Annoy index
        annoy_index.add_item(i, output_tensor)

        # Log progress
        if i % 100 == 0:
            print(f'Processed {i}/{len(images)} images')
    except Exception as e:
        print(f"Error processing image {image_name}: {e}")

# Build and save Annoy index
annoy_index.build(10)
annoy_index.save('Cat_index.ann')

print("Index built and saved successfully.")
