import os
import json
import torch
import gdown
from itertools import product
from PIL import Image
from torchvision import transforms
from flask_sqlalchemy import SQLAlchemy
from siamese_network import SiameseNetwork
from database import db, ImageModel, RecommendationResult
from tqdm import tqdm  # make sure you have tqdm installed

# ✅ Step 1: Download model if missing
MODEL_PATH = "siamese_model.pt"
GOOGLE_DRIVE_ID = "1b33sVOOrKvb7fFQieD-oMW7Q41hckbBD"  # Replace with your actual file ID

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}", MODEL_PATH, quiet=False)
        print("✅ Model downloaded!")

download_model_if_needed()

# ✅ Step 2: Setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def create_blank_tensor():
    blank_image = PILRaw.new("RGB", (224, 224), (255, 255, 255))  # White blank PIL image
    tensor_img = transform(blank_image).unsqueeze(0).to(device)
    return tensor_img, blank_image


def generate_recommendations(user_id):
    """
    Generate, predict, and save outfit combinations based on the user's uploaded images.
    Save into RecommendationResult (prediction) and GeneratedOutfit (pure combinations).
    """
    print(f"🔄 Generating outfit combinations for user: {user_id}")

    # 1️⃣ Fetch images and bucket by category
    user_images = ImageModel.query.filter_by(user_id=user_id).all()
    category_mapping = {}
    for img in user_images:
        category_mapping.setdefault(img.category, []).append(img.image_path)

    tops = category_mapping.get("Tops", [])
    bottoms = category_mapping.get("Bottoms", [])
    shoes = category_mapping.get("Shoes", [])
    allwear = category_mapping.get("All-wear", [])
    optional_categories = {
        k: v for k, v in category_mapping.items()
        if k not in ["Tops", "Bottoms", "Shoes", "All-wear"]
    }
    optional_values = list(optional_categories.values())

    # 2️⃣ Build all raw combinations
    valid_combinations = []

    for r in range(2, 8):
        # Flow 1: Tops + Bottoms + Shoes
        if tops and bottoms and shoes:
            base = [tops, bottoms, shoes]
            n = r - len(base)
            if n == 0:
                valid_combinations += list(product(*base))
            elif n > 0:
                for opt_comb in combinations(optional_values, n):
                    slots = base + list(opt_comb)
                    valid_combinations += list(product(*slots))

        # Flow 2: All-wear + Shoes
        if allwear and shoes:
            base = [allwear, shoes]
            n = r - len(base)
            if n == 0:
                valid_combinations += list(product(*base))
            elif n > 0:
                for opt_comb in combinations(optional_values, n):
                    slots = base + list(opt_comb)
                    valid_combinations += list(product(*slots))

    if not valid_combinations:
        print("⚠️ No valid combinations found")
        return []

    # 3️⃣ Filter combinations that mix All-wear + Outerwear
    allwear_set = set(category_mapping.get("All-wear", []))
    outerwear_set = set(category_mapping.get("Outerwear", []))

    filtered_combinations = [
        combo for combo in valid_combinations
        if not (any(item in allwear_set for item in combo) and
                any(item in outerwear_set for item in combo))
    ]

    if not filtered_combinations:
        print("⚠️ All combinations with All-wear + Outerwear have been removed")
        return []

    # 4️⃣ Predict and Save
    print(f"\n🧥 Generated {len(filtered_combinations)} outfit combinations:\n")

    upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))

    for idx, combo in enumerate(tqdm(filtered_combinations, desc="🔍 Predicting outfits"), start=1):
        outfit_files = list(combo)
        category_filename_pairs = []

        for item in outfit_files:
            found_category = next((cat for cat, items in category_mapping.items() if item in items), None)
            if found_category:
                mapped_cat = "All-body/Tops" if found_category in ["All-body", "Tops"] else found_category
                category_filename_pairs.append((mapped_cat, item))

        new_generated_outfit = GeneratedOutfit(
            user_id=user_id,
            outfit=json.dumps(outfit_files)
        )
        db.session.add(new_generated_outfit)

        slot_order = [
            "Hats", "Accessories", "Sunglasses",
            "Outerwear", "All-body/Tops", "Bottoms", "Shoes"
        ]
        
        CATEGORY_RENAME = {
            "All-body": "All-body/Tops",
            "Tops": "All-body/Tops",
            "All-wear": "All-body/Tops"
        }

        category_slot_mapping = {}
        for cat, fname in category_filename_pairs:
            mapped_cat = CATEGORY_RENAME.get(cat, cat)
            category_slot_mapping[mapped_cat] = fname

        input_batch = []
        for slot in slot_order:
            if slot in category_slot_mapping:
                img_path = os.path.join(upload_dir, category_slot_mapping[slot])
                img = PILImage.open(img_path).convert("RGB")
                tensor_img = transform(img).unsqueeze(0).to(device)
                input_batch.append(tensor_img)
            else:
                blank_tensor, _ = create_blank_tensor()
                input_batch.append(blank_tensor)

        model.eval()
        with torch.no_grad():
            logits, *_ = model(*input_batch)
            probs = torch.softmax(logits[0], dim=0).cpu().numpy()

        scores_dict = {EVENT_LABELS[i]: float(probs[i]) for i in range(len(EVENT_LABELS))}
        top_event_idx = probs.argmax()
        top_event = EVENT_LABELS[top_event_idx]
        top_score = float(probs[top_event_idx])

        result = RecommendationResult(
            user_id=user_id,
            event=top_event,
            outfit=json.dumps(outfit_files),
            scores=json.dumps(scores_dict),
            match_score=top_score,
            heatmap_paths="[]"
        )
        db.session.add(result)

    db.session.commit()
    print(f"✅ Saved {len(filtered_combinations)} generated outfits and prediction results for user {user_id}.")

    return filtered_combinations
