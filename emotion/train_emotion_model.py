import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from emotion_cnn import EmotionCNN  # Import your CNN model class

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset and DataLoader
train_dataset = datasets.ImageFolder("emotion_dataset/train", transform=transform)
val_dataset = datasets.ImageFolder("emotion_dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = EmotionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx % 100 == 0:  # Print every 100 batches
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {running_loss/100}")
            running_loss = 0.0
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} completed.")


# Save the model
torch.save(model.state_dict(), 'emotion_model.pth')
