import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from functions import get_device, build_model, save_checkpoint, data_loader

from tqdm import tqdm


def model_training(model, train_loader, valid_loader, criterion, optimizer, epochs, device):
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        
        train_accuracy = 0.0
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            top_class = outputs.argmax(dim=1)
            equals = top_class == labels
            train_accuracy += equals.float().mean().item()
                    
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_accuracy = 0.0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                
                top_class = outputs.argmax(dim=1)
                equals = top_class == labels
                valid_accuracy += equals.float().mean().item()
        
        # Calculate average losses and validation accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = train_accuracy / len(train_loader)
        val_loss = valid_loss / len(valid_loader)
        val_acc = valid_accuracy / len(valid_loader)
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {train_loss:.3f}.. "
              f"Train accuracy: {train_acc:.3f}.. "
              f"Validation loss: {val_loss:.3f}.. "
              f"Validation accuracy: {val_acc:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Training a neural network on images of flowers')
    parser.add_argument('data_dir', type=str, default="flowers/", help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['resnet18', 'resnet50','vgg16','vgg19'], help='Model architecture')
    parser.add_argument('--hidden_units', type=int, default=512, 
                        help='Number of hidden units (default: 512)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate ')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU for training')
    args = parser.parse_args()

    train_loader, valid_loader, _ = data_loader(args.data_dir)

    device = get_device(args.gpu)
    
    architecture = args.arch
    output_size=len(train_loader.dataset.classes)

    model = build_model(architecture, args.hidden_units, output_size ,args.dropout)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model_training(model, train_loader, valid_loader, criterion, optimizer, args.epochs, device)
    
    class_to_idx=train_loader.dataset.class_to_idx
    
    save_checkpoint(model, architecture, class_to_idx, args.save_dir,
                    args.hidden_units, args.dropout)


if __name__ == '__main__':
    main()


