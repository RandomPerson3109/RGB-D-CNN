import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class YourCustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.image_paths = os.listdir(data_folder)
        # Optionally, you can load labels or any other necessary data here

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.image_paths[idx])
        image = Image.open(img_name).convert('RGB')
        # Optionally, load corresponding labels here if available

        if self.transform:
            image = self.transform(image)

        # Return image and label (if applicable)
        return image

# Define transformations
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 1. Dataset Preparation
# Define your custom dataset class and preprocessing steps

# 2. Data Loading
train_dataset = YourCustomDataset("C:/Users/최신우/PycharmProjects/CNN/Train", transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = YourCustomDataset("C:/Users/최신우/PycharmProjects/CNN/Test", transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32)

# 3. GoogLeNet Model
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', weights=None)
# Modify GoogLeNet if needed (e.g., change the number of classes in the final fully connected layer)

# 4. Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Validation or testing after each epoch
    model.eval()
    # Validation/testing loop here

# 5. Evaluation
# Evaluate the model on the test dataset
