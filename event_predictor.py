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
    blank_image = PILRaw.new("RGB", (224, 224), (255, 255, 255))
    tensor_img = transform(blank_image).unsqueeze(0).to(device)
    return tensor_img, blank_image

def tensor_to_base64(tensor):
    """
    Directly converts tensor (C, H, W) to a grid image (first 5 channels) without using matplotlib.
    """
    np_tensor = tensor.detach().cpu().numpy()  # shape: (C, H, W)
    channels = np_tensor[:5]  # first 5 channels

    # Normalize each channel individually
    channel_images = []
    for ch in channels:
        ch_min, ch_max = ch.min(), ch.max()
        ch_norm = (ch - ch_min) / (ch_max - ch_min + 1e-5)  # normalize to [0,1]
        ch_img = (ch_norm * 255).astype(np.uint8)
        img = PILImage.fromarray(ch_img).convert("L").resize((224, 224))
        channel_images.append(img)

    # Create a combined horizontal image
    total_width = 224 * len(channel_images)
    combined_img = PILImage.new('RGB', (total_width, 224))
    for idx, img in enumerate(channel_images):
        combined_img.paste(img.convert("RGB"), (idx * 224, 0))

    # Save into base64
    buf = io.BytesIO()
    combined_img.save(buf, format='PNG')
    buf.seek(0)
    image_bytes = buf.read()
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


def predict_event_from_filenames(category_filename_pairs):
    category_order = [
        "Hats", "Accessories", "Sunglasses",
        "Outerwear", "All-body/Tops", "Bottoms", "Shoes"
    ]

    CATEGORY_RENAME = {
        "All-body": "All-body/Tops",
        "Tops": "All-body/Tops"
    }

    category_map = {}
    for cat, fname in category_filename_pairs:
        mapped_cat = CATEGORY_RENAME.get(cat, cat)
        category_map[mapped_cat] = fname

    input_batch, raw_images = [], []
    for category in category_order:
        if category in category_map:
            filepath = os.path.join("uploads", category_map[category])
            img = PILImage.open(filepath).convert("RGB")
            tensor_img = transform(img).unsqueeze(0).to(device)
            input_batch.append(tensor_img)
            raw_images.append(img)
        else:
            blank_tensor, blank_img = create_blank_tensor()
            input_batch.append(blank_tensor)
            raw_images.append(blank_img)

    model.eval()
    with torch.no_grad():
        logits, fmap_list, masks, trace_outputs = model(*input_batch)

    # 🎯 Collect visualization data
    attention_maps = []
    for idx, trace in enumerate(trace_outputs):
        if idx >= len(input_batch):
            break
        if masks[idx] == 1:
            attn_map = {}
            if 'after_attn1' in trace:
                attn_map["layer1"] = tensor_to_base64(trace['after_attn1'][0])
            if 'after_attn2' in trace:
                attn_map["layer2"] = tensor_to_base64(trace['after_attn2'][0])
            if 'after_attn3' in trace:
                attn_map["layer3"] = tensor_to_base64(trace['after_attn3'][0])
            if 'after_attn4' in trace:
                attn_map["layer4"] = tensor_to_base64(trace['after_attn4'][0])
            attention_maps.append(attn_map)

    cross_attn_vecs = []
    global_attn_vecs = []

    for t in trace_outputs:
        if 'after_cross_attn' in t:
            cross_attn = t['after_cross_attn'][0]
            masks_np = masks.cpu().numpy()
            for idx in range(cross_attn.shape[0]):
                if masks_np[idx] == 1:
                    cross_attn_vecs.append(cross_attn[idx, :20].cpu().numpy().tolist())
        if 'after_global_attn' in t:
            global_attn = t['after_global_attn'][0]
            for idx in range(global_attn.shape[0]):
                if masks_np[idx] == 1:
                    global_attn_vecs.append(global_attn[idx, :20].cpu().numpy().tolist())

    probs = torch.softmax(logits[0], dim=0)
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
        ],
        'attention_maps': attention_maps,
        'cross_attention': cross_attn_vecs,
        'global_attention': global_attn_vecs
    }
