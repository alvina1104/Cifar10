

from fastapi import FastAPI, UploadFile, File, HTTPException, File
import io
import uvicorn
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image


labels = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


class CifarClassifaction(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),#32x32
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),#16x16
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),#8x8
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

  def forward(self, image):
    image = self.first(image)
    image = self.second(image)
    return image

transforms = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


cifar_app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CifarClassifaction()
model.load_state_dict(torch.load("cifar_model.pth", map_location=device))
model.to(device)
model.eval()


@cifar_app.post('/predict/')
async def cifar_img(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail='Image is empty')

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        img_tensor = transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = y_pred.argmax(dim=1).item()


        predicted_label = labels[pred]

        return {
            "class_id": pred,
            "label": predicted_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == '__main__':
    uvicorn.run(cifar_app, host="127.0.0.1", port=8000)
