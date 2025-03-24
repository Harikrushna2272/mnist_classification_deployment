import torch
from PIL import Image
from torchvision.transforms import transforms
import torch.nn as nn
from io import BytesIO  # ✅ Import BytesIO

# Load model
class MNISTNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inh1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.h2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.h3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.h4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.h5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.inh1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.h2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.h3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.h4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.h5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Load model
model = MNISTNN()
PATH = 'model_mnist.pth'
state_dict = torch.load(PATH, map_location=torch.device('cpu'))

# Remove unexpected keys
for key in list(state_dict.keys()):
    if "bn6" in key:
        del state_dict[key]

# Load modified state dict
model.load_state_dict(state_dict, strict=False)
model.eval()


# Convert image to tensor
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # ✅ Fixed typo
        transforms.Resize((28,28)),  # ✅ Fixed typo
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # ✅ Fixed standard deviation
    ])
    image = Image.open(BytesIO(image_bytes))
    return transform(image).unsqueeze(0).view(-1, 28*28)  # ✅ Flatten to match input size

# Get model prediction
def get_prediction(image_tensor):
    output = model(image_tensor)  # ✅ Fixed incorrect variable
    prediction = torch.argmax(output, 1)
    return prediction
