import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import ssl
from fvcore.nn import FlopCountAnalysis

import warnings
warnings.filterwarnings("ignore")

# Step 1: Define transformations (Resizing and normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


root_path = 'Dataset'
train_iamge_path = 'Dataset/training'
test_image_path = 'Dataset/testing'

# Step 2: Load dataset using ImageFolder (assuming dataset is organized into class-specific directories)
train_dataset = datasets.ImageFolder(train_iamge_path, transform=transform)
test_dataset = datasets.ImageFolder(test_image_path, transform=transform)

# Step 3: Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 4: Load EfficientNet-B1 pretrained model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Disable SSL verification (use with caution, only for testing)
ssl._create_default_https_context = ssl._create_unverified_context

# # Step 2: Load the pre-trained EfficientNet-B1 model
model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT).to(device)

# Step 3: Modify the classifier to match the number of classes in your dataset
num_classes = len(train_dataset.classes)  # Assuming train_dataset is defined earlier

# # Step 4: Replace the last fully connected layer
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

# # Step 5: Move the model to the specified device
model = model.to(device)

# Step 6: Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 7: Calculate FLOPs and parameters -- Optional 
'''
try:
    from fvcore.nn import FlopCountAnalysis, parameter_count

    def calculate_flops(model, input_size=(1, 3, 224, 224)):
        dummy_input = torch.randn(*input_size).to(device)
        flops = FlopCountAnalysis(model, dummy_input)
        params = parameter_count(model)
        print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
        print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run FLOP calculation before training
    calculate_flops(model)

except ImportError:
    print("fvcore is not installed, skipping FLOPs calculation.")

'''

# ----- Uncomment the following train_model function if you want to train your own model -----


# Step 7: Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):

    # Set the model to training mode
    model.train()

    # Use the scheduler only if you want to adjust the learning rate during training.

    # Learning rate scheduler - This helps to adjust the learning rate during training
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, leave=True)
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()
            
            # Update tqdm loop
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=running_loss/len(train_loader), accuracy=100.*correct/total)
        
        # Step the learning rate scheduler
        # scheduler.step()
        # Print current learning rate for reference
        # current_lr = optimizer.param_groups[0]['lr']
        # print(f"Epoch [{epoch+1}/{num_epochs}], Current Learning Rate: {current_lr}")

        
    torch.save(model.state_dict(), 'fine_tuned_efficientnet_v2_s.pth')
    print(f"The Training has completed - Model saved")


# Step 8: Evaluation function
def evaluate_model(model, test_loader, criterion, device, model_path=None):
    
    # 
    if model_path:
        # Load the saved model - gpu - mps 
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path} for evaluation.")

    # following code might be necessary to load the model if you are using cpu or cuda
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda'))) - if your gpu is cuda.
    # if model_path:
    #     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    #     model.to(device)
    #     print(f"Model loaded from {model_path} for evaluation")
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Evaluating Model", leave=True)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()

            # Update tqdm with current batch loss and accuracy
            loop.set_postfix({
                'Validation Loss': f'{running_loss/len(test_loader):.4f}',
                'Accuracy': f'{100.*correct/total:.2f}%'
            })
    
    accuracy = 100. * correct / total
    loss = running_loss / len(test_loader)
    
    print(f"\nValidation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.2f}%")


# Step 9: Train and evaluate the model
num_epochs = 10

# train_model(model, train_loader, criterion, optimizer, device, num_epochs)

evaluate_model(model, test_loader, criterion, device, model_path=f'fine_tuned_efficientnet_v2_s.pth')