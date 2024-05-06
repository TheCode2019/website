import torch
import torchvision.transforms as transforms
from PIL import Image
from ocr import Enhanced_OCR_CNN  
import string

# Definition des index du dataset
label_map = list(string.digits) + list(string.ascii_uppercase) + list(string.ascii_lowercase)

def load_model(model_path):
    """
    Load a pre-trained model from the specified path.
    """
    model = Enhanced_OCR_CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image_path, model):
    """
    Predict the class index for an input image using the provided model.
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Normalisation du format voulu
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalisation en accordance avec l'entrainement
    ])
    image = Image.open(image_path).convert('L')  # Conversion grayscale
    image = transform(image)
    image = image.unsqueeze(0)  # batch dimension
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

def decode_label(label_index):
    """
    Fonction pour decoder l'index de la classe prédite en caractère.
    """
    if label_index < len(label_map):
        return label_map[label_index]
    else:
        return "Non reconnu par le modèle"

def main():
    model_path = 'model_ocr.pth'
    image_path = r'C:\Users\yassi\Desktop\English\Fnt\Sample024\img024-00008.png'
    
    model = load_model(model_path)
    prediction_index = predict(image_path, model)
    predicted_character = decode_label(prediction_index)
    
    print("Index prédit par le modèle:", prediction_index)
    print("Caractère prédit par le modèle:", predicted_character)

if __name__ == "__main__":
    main()
