'''
@brief Mobile-Net architecture, suitable for mobile processes.
@author Ayush Tripathi (atripathi7783@gmail.com)
'''


#imports
import torch
import torch.nn as nn
from src.back_end.model_training.arch.interface import *
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Any




'''
To Do:
- Use a pre-trained model from Image-Net 
- train on TACO afterwards 


DepthWiseSeperable is a class that decomposes typical convolution into depthwise and pointwise convolutions, requiring less resources as matrices are smaller in dimension. Adapted from Karunesh Upadhyay's description. 
'''

class DepthWiseSeperable(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        DepthWiseSeperable block for MobileNet.
        """
        super(DepthWiseSeperable, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, 
            kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MobileNetV1(nn.Module, CVModel):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        # Initial convolution layer
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Depthwise separable convolutions
            DepthWiseSeperable(32, 64, 1),
            DepthWiseSeperable(64, 128, 2),
            DepthWiseSeperable(128, 128, 1),
            DepthWiseSeperable(128, 256, 2),
            DepthWiseSeperable(256, 256, 1),
            DepthWiseSeperable(256, 512, 2),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 1024, 2),
            DepthWiseSeperable(1024, 1024, 1),
        )

        # Average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        #run through features and pool
        x = self.features(x)
        x = self.avgpool(x)

        #reshaping tensor to linear classifier
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x        


def train_model(self, train_loader: DataLoader, val_loader: DataLoader, **kwargs) -> dict[str, Any]:
    """
    Train the MobileNetV1 model on the provided dataset.
    
    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        **kwargs: Additional keyword arguments for training configuration.

    Returns:
        dict[str, Any]: Dictionary containing training history and final validation accuracy.
    """
    # Get hyperparameters and defaults
    num_epochs = kwargs.get("num_epochs", 10)
    lr = kwargs.get("lr", 0.01)
    weight_decay = kwargs.get("weight_decay", 1e-4)
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Define optimizer, loss function, and scheduler
    optimizer = kwargs.get("optimizer", torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay))
    criterion = kwargs.get("criterion", nn.CrossEntropyLoss())
    scheduler = kwargs.get("scheduler", torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1))
    
    # Move model to the appropriate device
    self.to(device)
    
    # Training history
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    
    # Training loop
    for epoch in range(num_epochs):
        self.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = self(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        # Validation phase
        val_loss, val_accuracy = self.validate_model(val_loader, criterion=criterion, device=device)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
        
        # Adjust learning rate
        scheduler.step()
    
    return history

def validate_model(self, loader: DataLoader, criterion: nn.Module, device: str = "cuda") -> tuple[float, float]:
    """
    Validate the model on the given dataset.

    Args:
        loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function for validation.
        device (str): Device to use for validation.

    Returns:
        tuple[float, float]: Validation loss and accuracy.
    """
    self.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = self(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(loader)
    val_accuracy = correct / total
    return avg_val_loss, val_accuracy
        

def test_model():
    pass

def predict():
    pass

def save():
    pass


if __name__ == "__main__":
    model = MobileNetV1(num_classes=7)  # 7 classes for TACO dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    history = model.train_model(
        train_loader=train_loader, 
        val_loader=val_loader, 
        num_epochs=15, 
        lr=0.001, 
        device="cuda"
    )