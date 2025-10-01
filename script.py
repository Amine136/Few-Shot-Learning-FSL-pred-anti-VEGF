import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image


#must be updated    
encoder_path = r'protonet_encoder_best_lr00015.pth'
support_set_path = r'support'  # must have class folders inside
#must be updated



# ----------- Encoder Definition -----------
class ConvEmbedding(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.encoder(x)

# ----------- Compute Prototypes -----------
def compute_prototypes(embeddings, labels):
    classes = torch.unique(labels)
    prototypes = []
    for c in classes:
        class_embeddings = embeddings[labels == c]
        prototype = class_embeddings.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes), classes

# ----------- Image Transform -----------
def preprocess_image(image_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension





encoder_path = r'C:\Users\HP\Desktop\few learning/protonet_encoder_best_lr00015.pth'
support_set_path = r'C:\Users\HP\Desktop\few learning/support'  # must have class folders inside

# 🧠 Load encoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = ConvEmbedding(out_dim=64)
encoder.load_state_dict(torch.load(encoder_path, map_location=device))
encoder.to(device)


# ----------- Inference Function -----------
def predict_image(image_path, encoder=encoder_path, support_path=support_set_path, device =device):
    # Preprocess the query image
    X_q = preprocess_image(image_path).to(device)

    # Load support set (required to compute class prototypes)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    support_set = ImageFolder(support_path, transform=transform)
    support_loader = torch.utils.data.DataLoader(support_set, batch_size=len(support_set), shuffle=False)
    X_s, y_s = next(iter(support_loader))
    X_s, y_s = X_s.to(device), y_s.to(device)

    # Embed both support and query
    with torch.no_grad():
        encoder.eval()
        emb_s = encoder(X_s)
        emb_s = F.adaptive_avg_pool2d(emb_s, 1).view(X_s.size(0), -1)

        emb_q = encoder(X_q)
        emb_q = F.adaptive_avg_pool2d(emb_q, 1).view(1, -1)

        prototypes, class_ids = compute_prototypes(emb_s, y_s)
        dists = torch.cdist(emb_q, prototypes)  # Shape: [1, n_classes]
        pred_idx = dists.argmin(dim=1).item()
        pred_class = support_set.classes[class_ids[pred_idx].item()]
        pred_result = "Good responders" if pred_class=="class_0" else 'Bad Responders'

    print(f"🖼️ Prediction for image '{image_path}': {pred_result}")
    return pred_result

# ----------- Run the Code -----------
if __name__ == '__main__':
    #Path

    query_image_path = r'C:\Users\HP\Desktop\few learning/bad.jpg'  # Change this to your actual image path



    #Predict
    predict_image(query_image_path, encoder, support_set_path, device)
