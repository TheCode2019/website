import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from ocr import Enhanced_OCR_CNN, Chars74KDataset  

def train_and_validate(model, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs=25, device="cuda"):
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        model.eval()
        validation_loss = 0
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}: Training Loss {training_loss/len(train_loader)}, Validation Loss {validation_loss/len(test_loader)}, Accuracy {(100*correct/total):.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Load OCR Model")
    parser.add_argument('--train', action='store_true', help="Specify this flag to train the model")
    args = parser.parse_args()

    if args.train:
        transformations = transforms.Compose([...])  
        dataset_directory = r'C:\Users\yassi\Desktop\English\Fnt'
        dataset = Chars74KDataset(directory=dataset_directory, transform=transformations)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Enhanced_OCR_CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        # Entrainer le modèle
        train_and_validate(model, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs=20, device=device)
        # Sauvegarder le modèle
        torch.save(model.state_dict(), 'model_ocr.pth')
