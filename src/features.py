import torch
import torchvision.transforms as T
from torchvision.models import resnet18

class FeatureExtractor:
    def __init__(self):
        self.model = resnet18(pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
        ])

    def extract(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        tensor = self.transform(crop).unsqueeze(0)
        with torch.no_grad():
            features = self.model(tensor).squeeze().numpy()
        return features
