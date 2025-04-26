import torch
from torchvision import transforms
from PIL import Image as PILImage, Image as PILRaw
import os
import gdown
from siamese_network import SiameseNetwork

# ✅ Google Drive Model Setup
MODEL_PATH = "siamese_model.pt"
GOOGLE_DRIVE_ID = "1b33sVOOrKvb7fFQieD-oMW7Q41hckbBD"  # Your model file ID

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}", MODEL_PATH, quiet=False)
        print("✅ Model downloaded!")

# ✅ Load the model with download logic
download_model_if_needed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ✅ Event labels
EVENT_LABELS = [
    "Job Interviews", "Birthday", "Graduations", "MET Gala", "Business Meeting",
    "Beach", "Picnic", "Summer", "Funeral", "Romantic Dinner",
    "Cold", "Casual", "Wedding"
]

# ✅ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def create_blank_tensor():
    blank = PILRaw.new("RGB", (224, 224), (255, 255, 255))
    return transform(blank).unsqueeze(0).to(device), blank

def predict_event_from_filenames(category_filename_pairs):
    category_order = [
        "Hats", "Accessories", "Sunglasses",
        "Outerwear", "All-body/Tops", "Bottoms", "Shoes"
    ]

    category_map = {cat: fname for cat, fname in category_filename_pairs}

    print("\n===========================\n📌 SLOT DEBUG: Model Input Order\n===========================")
    for idx, category in enumerate(category_order):
        status = f"🟢 Using {category_map[category]}" if category in category_map else "⬜ Blank Tensor"
        print(f"[{idx}] {category:<22} → {status}")

    input_batch = []
    for category in category_order:
        if category in category_map:
            filepath = os.path.join("uploads", category_map[category])
            img = PILImage.open(filepath).convert("RGB")
            tensor_img = transform(img).unsqueeze(0).to(device)
            input_batch.append(tensor_img)
        else:
            blank_tensor, _ = create_blank_tensor()
            input_batch.append(blank_tensor)

    model.eval()
    with torch.no_grad():
        logits, _, _, _ = model(*input_batch)

        probs = torch.softmax(logits[0], dim=0)  # Shape: (13,)

        prob_list = list(zip(EVENT_LABELS, probs.cpu().tolist()))
        prob_list.sort(key=lambda x: x[1], reverse=True)

        return {
            'top_3_predictions': [
                {'event': label, 'probability': round(p, 4)}
                for label, p in prob_list[:3]
            ],
            'all_event_probabilities': [
                {'event': label, 'probability': round(p, 4)}
                for label, p in prob_list
            ]
        }
