from fastapi import FastAPI, File, UploadFile
from torchvision import models, transforms
from PIL import Image
import torch
import io

import pickle
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Load the trained model
model = models.resnet18(pretrained=False)  # Load without pre-trained weights
num_classes = 3  # Bungalow, Highrise, Storey-Building
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
with open('./model/model_ft.pkl', 'rb') as f:
    model = pickle.load(f)
model.eval()

# Define class names
class_names = ['Bungalow', 'Highrise', 'Storey-Building']

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Accepts an image file and returns the predicted building type."""
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]
    
    return {"prediction": label}

