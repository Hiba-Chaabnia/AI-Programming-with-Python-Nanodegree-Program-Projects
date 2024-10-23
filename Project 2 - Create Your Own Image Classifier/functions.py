import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim , nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from PIL import Image


from torchvision import datasets , transforms , models  

def data_loader(data_dir="./flowers"):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'train':transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        
        'val': transforms.Compose([transforms.RandomResizedCrop(256),transforms.CenterCrop(224),
                                   transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        
        'test': transforms.Compose([transforms.RandomResizedCrop(256),transforms.CenterCrop(224),
                                    transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }
    
    
    image_datasets = {
        'train' : datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        'val' : datasets.ImageFolder(valid_dir, transform=data_transforms["val"]),
        'test' : datasets.ImageFolder(test_dir, transform=data_transforms["test"])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True ,num_workers=2, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=32, num_workers=2, pin_memory=True),
        'test': DataLoader(image_datasets['test'], batch_size=32)
    }
    
    train_loader = dataloaders['train']
    valid_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    return train_loader, valid_loader, test_loader


def get_device(use_gpu):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def define_classifier(input_size, hidden_units=512, output_size=102, dropout_rate=0.5):
    return nn.Sequential(
        nn.Linear(input_size, hidden_units), 
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_units, output_size),  
        nn.LogSoftmax(dim=1)
    )


def build_model(architecture='vgg16', hidden_units=512 ,output_size=102, dropout_rate=0.5):
    
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
        model.classifier = define_classifier(input_size, hidden_units, output_size, dropout_rate)
    
    elif architecture == 'vgg19':  
        model = models.vgg19(pretrained=True)
        input_size = model.classifier[0].in_features
        model.classifier = define_classifier(input_size, hidden_units, output_size, dropout_rate)

    elif architecture == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = model.fc.in_features 
        model.fc = define_classifier(input_size, hidden_units, output_size, dropout_rate)

    elif architecture == 'resnet50':  
        model = models.resnet50(pretrained=True)
        input_size = model.fc.in_features  
        model.fc = define_classifier(input_size, hidden_units, output_size, dropout_rate)

    else:
        raise ValueError(f"This architecture {architecture} is not supported."
                         f"Supported architectures are : vgg16 , vgg19 , resnet18 and resnet50.")
    
    return model


def save_checkpoint(model, architecture, class_to_idx, checkpoint_path, hidden_units, dropout_rate):
    checkpoint = {
        'architecture': architecture,
        'hidden_units': hidden_units,
        'dropout_rate': dropout_rate,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }
    torch.save(checkpoint, checkpoint_path)

    

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    architecture = checkpoint['architecture']
    hidden_units = checkpoint['hidden_units']  
    dropout_rate = checkpoint['dropout_rate']  
    class_to_idx = checkpoint['class_to_idx']
    
    output_size = len(class_to_idx)  
    
    model = build_model(architecture, hidden_units, output_size, dropout_rate)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    model.class_to_idx = class_to_idx

    return model, class_to_idx



def process_image(image_path):

    image = Image.open(image_path)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    return preprocess(image)


def imshow(image, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo normalization
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    

    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    if title:
        ax.set_title(title)
    
    return ax

def predict(image_path, model, top_k=5, device='cpu'):

    model.to(device)
    model.eval()
    
    tensor_image = process_image(image_path)
    tensor_image = tensor_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_image)
        
    probabilities = torch.softmax(output, dim=1)
    top_probs, top_indices = probabilities.topk(top_k, dim=1)
    
    top_probs = top_probs.squeeze().cpu().numpy()  
    top_indices = top_indices.squeeze().cpu().numpy()
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_indices]
    
    return top_probs, top_classes

def load_cat_names(json_path):

    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name




