import torch
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

#bad attempt at exporting as ONNX

sorted_set = Path("sorted")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

img_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset_unsplit = datasets.ImageFolder(
    root=sorted_set,
    transform=img_transform
)

train_set, test_set = random_split(dataset_unsplit, [0.8, 0.2])

class GUPPYmyelomaV1(nn.Module):
    def __init__(self, hidden_units_model: int = 16, dropout_p: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, hidden_units_model, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_units_model, hidden_units_model * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(hidden_units_model * 2, hidden_units_model * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((hidden_units_model * 2) * 8 * 8, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


hidden_units = 10
model = GUPPYmyelomaV1(hidden_units_model=hidden_units)

state = torch.load("skinmodel.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

dummy = torch.randn(1, 3, 64, 64)

torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18
)
