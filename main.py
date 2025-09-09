import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()

def resnet18_model(model_path, device):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def densenet121_model(model_path, device):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def cnn_model(model_path, device):
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 16 * 16, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATHS = {
    "resnet18": r"C:\Users\ihhim\OneDrive\Desktop\project1\models\resnet18.pth",
    "densenet121": r"C:\Users\ihhim\OneDrive\Desktop\project1\models\best_model.pth",
    "cnn_model": r"C:\Users\ihhim\OneDrive\Desktop\project1\models\cnn_model.pth"
}
MODEL_LOADERS = {
    "resnet18": resnet18_model,
    "densenet121": densenet121_model,
    "cnn_model": cnn_model
}

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    image_bytes = await file.read()
    if model_name == "densenet121":
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    else:
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
    img_tensor = transforms.ToTensor()(image).unsqueeze(0)
    model = MODEL_LOADERS[model_name](MODEL_PATHS[model_name], device)
    model = model.to(device)
    
    with torch.no_grad():
        output = model(img_tensor.to(device))
        pred = torch.argmax(output, dim=1).item()

    class_names = ["NORMAL", "PNEUMONIA"]
    return {"prediction": class_names[pred]}