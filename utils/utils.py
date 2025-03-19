import wandb
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from models.models import *
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from torchvision.transforms import Lambda
import torchaudio.transforms as T


# Custom dataset class to handle image data loading
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe # Store the DataFrame
        self.transform = transform # Store the transform (if any)

    def __len__(self):
        return len(self.dataframe) # Return the size of the dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load the image from file
        img_path = self.dataframe.iloc[idx]['image_paths']
        image = img_to_array(load_img(img_path, target_size=(224, 224)))  # Shape (224, 224, 3) (height, width, channels)
        label = self.dataframe.iloc[idx]['labels']
        other = self.dataframe.iloc[idx]['other_labels']  # Assuming 'other_labels' column exists in your dataframe
        
        if self.transform: # Automatic transformation aplication
            image = self.transform(image) # Apply transformations if any
        else: # Manual conversion to tensor and reshaping
            image = torch.tensor(image).permute(2, 0, 1).float()  # Convert to shape (3, 224, 224) 3 channels and size. PyTorch expects image tensors in the shape (channels, height, width).
        
        return image, torch.tensor(label).long(), other  # NOW THE IMAGES, LABELS, AND OTHER LABELS ARE RETURNED
    

def get_data(dataframe, train=True):
    train_data, test_data = train_test_split(dataframe, test_size=0.2, random_state=42)
    return train_data if train else test_data

def get_train_val_data(dataframe, val_size=0.1):
    train_data, val_data = train_test_split(dataframe, test_size=val_size, random_state=42)
    return train_data, val_data

def make_loader(dataset, batch_size):
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    return loader

def make(config, dataframe, device="cuda"):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts png image (original) to PyTorch tensor in CustomImageDataset using PIL library even if the image is png and not pil
    ])

    # Split data into training, validation, and test sets
    train_data, val_data = get_train_val_data(dataframe, val_size=0.1)
    train_data, test_data = get_train_val_data(train_data, val_size=0.2)  # Split training data into train and test

    # Create datasets for training, validation, and test
    train_dataset = CustomImageDataset(train_data, transform=transform)
    val_dataset = CustomImageDataset(val_data, transform=transform)
    test_dataset = CustomImageDataset(test_data, transform=transform)

    # Create data loaders for training, validation, and test
    train_loader = make_loader(train_dataset, batch_size=config['batch_size'])
    val_loader = make_loader(val_dataset, batch_size=config['batch_size'])
    test_loader = make_loader(test_dataset, batch_size=config['batch_size'])

    # Initialize the model and define loss function and optimizer
    # DEFINEN THERE THE MODEL YOU WANT TO USE
    model = ImprovedMusicGenreCNNv2(num_classes=config['classes']).to(device) #ConvNet:config['kernels']  #ResNet50 on ara hi ha custom model #ImprovedMusicGenreCNNv2 (config['classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5) #torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    return model, train_loader, val_loader, test_loader, criterion, optimizer

