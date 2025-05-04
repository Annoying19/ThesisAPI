from flask import Flask, request, jsonify, send_from_directory, abort
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import threading
from database import db, ImageModel, RecommendationResult, User, Saved, UploadedImage, GeneratedOutfit
from recommend_outfits import generate_recommendations
import json
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from PIL import Image
import json
from event_predictor import predict_event_from_filenames
# ✅ Initialize Flask App
app = Flask(__name__)


CORS(app)  # Enable CORS for React Native

# ✅ Ensure Database & Uploads Folder Exists
os.makedirs("assets", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# ✅ SQLite3 Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.abspath("assets/database.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)



# Load once globally or in your init
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')  # adjust path if needed

with open(CONFIG_PATH) as f:
    config = json.load(f)

API_URL = config['API_URL']


bcrypt = Bcrypt(app)    
db.init_app(app)
with app.app_context():
    db.create_all()

import io
# UPLOADS IMAGES IN VSCode
@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    print("Looking for file:", file_path)
    if not os.path.exists(file_path):
        print("❌ File not found!")
        return abort(404)
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# REGISTERS USERS
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data or "username" not in data or "password" not in data or "security_question" not in data or "security_answer" not in data:
        return jsonify({"error": "Missing required fields"}), 400

    username = data["username"]
    password = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
    security_question = data["security_question"]
    security_answer = data["security_answer"]

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400

    new_user = User(
        username=username,
        password=password,
        security_question=security_question,
        security_answer=security_answer
    )
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully!"}), 201

# LOGIN USERS
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or "username" not in data or "password" not in data:
        return jsonify({"error": "Missing required fields"}), 400   

    username = data["username"]
    password = data["password"]

    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password, password):
        return jsonify({"message": "Login successful", "user_id": user.id}), 200
    else:
        return jsonify({"error": "Invalid username or password"}), 401

@app.route('/get_user_info', methods=['GET'])
def get_user_info():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400

    user = User.query.filter_by(id=user_id).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'username': user.username})

@app.route("/get_security_question", methods=["POST"])
def get_security_question():
    data = request.get_json()
    username = data.get("username")

    if not username:
        return jsonify({"error": "Missing username"}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"question": user.security_question}), 200


@app.route("/reset_password", methods=["POST"])
def reset_password():
    data = request.get_json()
    username = data.get("username")
    answer = data.get("answer")
    new_password = data.get("new_password")

    if not username or not answer or not new_password:
        return jsonify({"error": "Missing fields"}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    if user.security_answer.strip().lower() != answer.strip().lower():
        return jsonify({"error": "Incorrect answer"}), 403

    user.password = bcrypt.generate_password_hash(new_password).decode("utf-8")
    db.session.commit()

    return jsonify({"message": "Password reset successfully!"}), 200

@app.route("/upload-multiple", methods=["POST"])
def upload_multiple_images():
    try:
        if "images" not in request.files or "user_id" not in request.form or "category" not in request.form:
            return jsonify({"error": "Missing required fields"}), 400

        user_id = int(request.form["user_id"])
        category = request.form["category"]

        user = db.session.get(User, user_id)
        if not user:
            return jsonify({"error": "Invalid user ID"}), 400

        category_prefix = {
            "Tops": "TOP",
            "Bottoms": "BTM",
            "Shoes": "SHO",
            "Outerwear": "OUT",
            "All-wear": "ALL",
            "Accessories": "ACC",
            "Hats": "HAT",
            "Sunglasses": "SUN"
        }

        category_code = category_prefix.get(category, "GEN")

        # Find max index used by this user in this category
        existing_ids = ImageModel.query.with_entities(ImageModel.id)\
                                       .filter_by(user_id=user_id, category=category).all()
        existing_numbers = []
        for (eid_str,) in existing_ids:
            try:
                num_part = eid_str.split(f"{category_code}")[-1]
                if num_part.isdigit():
                    existing_numbers.append(int(num_part))
            except:
                continue

        start_number = max(existing_numbers, default=0) + 1

        uploaded_images = []
        images = request.files.getlist("images")
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

        for idx, image in enumerate(images):
            unique_number = start_number + idx
            image_id = f"U{user_id}_{category_code}{unique_number:02d}"
        
            base_name = secure_filename(image.filename).rsplit('.', 1)[0]
            filename = f"{uuid.uuid4().hex}_{base_name}.png"  # 🔁 Save as .png
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
            image_bytes = image.read()
            if len(image_bytes) < 1000:
                return jsonify({"error": "Uploaded image is too small or empty."}), 400
        
            # ✅ Load as PIL Image and convert to RGB to preserve consistent pixel format
            try:
                from PIL import Image
                from io import BytesIO
                image_obj = Image.open(BytesIO(image_bytes)).convert("RGB")
                image_obj.save(save_path, format="PNG")  # 🔁 Save as PNG (lossless)
                # 🧪 Debug: Check if image is gray after saving
                reloaded_img = Image.open(save_path).convert("RGB")
                pixels = reloaded_img.getdata()
                sample_pixels = list(pixels)[:10]  # Check the first 10 pixels
                
                print(f"✅ Image saved to: {save_path}")
                print(f"🧪 Sample pixel values: {sample_pixels}")
                print("✅ Image saved at:", save_path)
            except Exception as e:
                print(f"❌ Failed to process image: {str(e)}")
                return jsonify({"error": "Image processing failed."}), 500
        
            new_image = ImageModel(
                id=image_id,
                image_path=filename,
                category=category,
                user_id=user_id
            )
            db.session.add(new_image)
            uploaded_images.append({
                "image_id": image_id,
                "image_path": f"{API_URL}/uploads/{filename}"
            })



        db.session.commit()

        def run_with_context(uid):
            with app.app_context():
                generate_recommendations(uid)

        background_thread = threading.Thread(target=run_with_context, args=(user_id,))
        background_thread.start()

        return jsonify({
            "message": "Images uploaded successfully! Recommendations are being generated.",
            "images": uploaded_images
        }), 201

    except Exception as e:
        print(f"❌ Flask Image Upload Error: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500



import os, json

# DELETE SELECTED IMAGES (cascade-remove Saved, Recommendations – and delete files)
@app.route("/delete-images", methods=["POST"])
def delete_images():
    data = request.get_json()
    image_ids = data.get("image_ids") or []
    if not image_ids:
        return jsonify({"error": "No images selected"}), 400

    # 1️⃣ Load all ImageModel instances to be deleted
    images = ImageModel.query.filter(ImageModel.id.in_(image_ids)).all()
    filenames = [img.image_path for img in images]

    # 2️⃣ Delete RecommendationResult rows
    recs_deleted = 0
    for rec in RecommendationResult.query.all():
        outfit = json.loads(rec.outfit)
        if any(os.path.basename(fn) in outfit for fn in filenames):
            db.session.delete(rec)
            recs_deleted += 1

    # 3️⃣ Delete Saved outfits
    saved_deleted = 0
    for saved in Saved.query.all():
        ids = saved.clothes_ids
        ids = ids if isinstance(ids, list) else (json.loads(ids) if ids else [])
        if any(str(i) in image_ids for i in ids):
            db.session.delete(saved)
            saved_deleted += 1

    # 4️⃣ Delete files on disk and remove ImageModel rows
    files_deleted = 0
    for img in images:
        path = os.path.join(app.config["UPLOAD_FOLDER"], img.image_path)
        if os.path.exists(path):
            os.remove(path)
            files_deleted += 1
        db.session.delete(img)

    db.session.commit()

    return jsonify({
        "message": (
            f"Deleted {len(images)} image record(s), removed {saved_deleted} saved outfit(s), "
            f"{recs_deleted} recommendation(s), and deleted {files_deleted} file(s) from uploads."
        )
    }), 200


# DELETE ALL IMAGES IN A CATEGORY (cascade-remove Saved, Recommendations – and delete files)
@app.route("/delete-all/<category>", methods=["DELETE"])
def delete_all_images(category):
    # 1️⃣ Gather all images in this category
    images = ImageModel.query.filter_by(category=category).all()
    if not images:
        return jsonify({"message": f"No images found in '{category}'."}), 200

    image_ids = [str(img.id) for img in images]
    filenames = [img.image_path for img in images]

    # 2️⃣ Delete RecommendationResult rows
    recs_deleted = 0
    for rec in RecommendationResult.query.all():
        outfit = json.loads(rec.outfit)
        if any(os.path.basename(fn) in outfit for fn in filenames):
            db.session.delete(rec)
            recs_deleted += 1

    # 3️⃣ Delete Saved outfits
    saved_deleted = 0
    for saved in Saved.query.all():
        ids = saved.clothes_ids
        ids = ids if isinstance(ids, list) else (json.loads(ids) if ids else [])
        if any(str(i) in image_ids for i in ids):
            db.session.delete(saved)
            saved_deleted += 1

    # 4️⃣ Delete files on disk and remove ImageModel rows
    files_deleted = 0
    for img in images:
        path = os.path.join(app.config["UPLOAD_FOLDER"], img.image_path)
        if os.path.exists(path):
            os.remove(path)
            files_deleted += 1
        db.session.delete(img)

    db.session.commit()

    return jsonify({
        "message": (
            f"All {len(images)} image record(s) in '{category}' deleted, "
            f"{saved_deleted} saved outfit(s), {recs_deleted} recommendation(s) removed, "
            f"and {files_deleted} file(s) cleaned up from uploads."
        )
    }), 200


# GETS THE CLOTHES BY CATEGORY
@app.route("/images/<category>", methods=["GET"])
def get_images_by_category(category):
    user_id = request.args.get('user_id')  # Get user_id from query params
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
    
    # Filter images by category and user_id
    images = ImageModel.query.filter_by(category=category, user_id=user_id).all()

    # Format the response
    image_list = [
        {
            "id": img.id,
            "image_path": f"{API_URL}/uploads/{img.image_path}",
            "category": img.category
        }
        for img in images
    ]
    
    return jsonify(image_list), 200



# GETS ALL CLOTHINGS OF USER
@app.route("/images/user/<user_id>", methods=["GET"])
def get_user_images(user_id):
    images = ImageModel.query.filter_by(user_id=user_id).all()  # Filter images by user_id
    return jsonify([{
        "id": img.id,
        "image_path": f"{API_URL}/uploads/{img.image_path}",
        "category": img.category
    } for img in images])

@app.route("/recommend", methods=["POST"])
def recommend_outfit():
    try:
        data = request.get_json()
        event = data.get("event")
        user_id = data.get("user_id")

        if not event or not user_id:
            return jsonify({"error": "Missing event or user ID"}), 400

        threshold = 0.20

        # 1️⃣ DL-based matches
        user_recommendations = RecommendationResult.query.filter_by(user_id=user_id).all()
        filtered_outfits = []
        for rec in user_recommendations:
            event_scores = json.loads(rec.scores)
            score = event_scores.get(event)
            if score and score >= threshold:
                filenames = json.loads(rec.outfit)
                image_urls = [f"{API_URL}/uploads/{fn}" for fn in filenames]
                filtered_outfits.append({
                    "match_score": score,
                    "outfit": image_urls,
                    "raw_filenames": filenames,
                    "scores": event_scores
                })

        # 2️⃣ FP-Growth event-based frequent patterns
        saved_outfits = Saved.query.filter_by(user_id=user_id).all()
        transactions = []
        event_lookup = {}

        for saved in saved_outfits:
            clothing_filenames = []
            for image_id in saved.clothes_ids:
                image = ImageModel.query.filter_by(id=image_id, user_id=int(user_id)).first()
                if image:
                    filename = os.path.basename(image.image_path)
                    clothing_filenames.append(filename)
            if clothing_filenames:
                transactions.append(clothing_filenames)
                event_lookup[tuple(sorted(clothing_filenames))] = saved.event.lower()

        frequent_event_outfits = []
        if transactions:
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df = pd.DataFrame(te_ary, columns=te.columns_)

            frequent_itemsets_df = fpgrowth(df, min_support=0.3, use_colnames=True)
            rules = association_rules(frequent_itemsets_df, metric="confidence", min_threshold=0.5)

            for _, row in rules.iterrows():
                itemset = row["antecedents"].union(row["consequents"])
                sorted_itemset = tuple(sorted(itemset))
                for trans, saved_event in event_lookup.items():
                    if set(sorted_itemset).issubset(set(trans)) and saved_event == event.lower():
                        image_urls = [f"{API_URL}/uploads/{img}" for img in sorted_itemset]
                        frequent_event_outfits.append({
                            "match_score": 0,
                            "boost_score": len(itemset),
                            "outfit": image_urls,
                            "raw_filenames": list(itemset),
                            "scores": {},
                            "support": row["support"],
                            "confidence": row["confidence"],
                            "lift": row["lift"]
                        })
                        break

        # 3️⃣ Boost DL-based results
        def boost_score(outfit_filenames):
            outfit_set = set(outfit_filenames)
            score = 0
            for fp in frequent_event_outfits:
                if set(fp["raw_filenames"]).issubset(outfit_set):
                    score += len(fp["raw_filenames"])
            return score

        for outfit in filtered_outfits:
            outfit["boost_score"] = boost_score(outfit["raw_filenames"])

        # 4️⃣ Merge results
        combined = filtered_outfits + frequent_event_outfits
        combined.sort(key=lambda x: (x["boost_score"], x["match_score"]), reverse=True)

        # Return empty results with 200 if no matches
        if not combined:
            return jsonify({
                "event": event,
                "results": []
            }), 200

        return jsonify({
            "event": event,
            "results": combined
        }), 200

    except Exception as e:
        print(f"❌ Recommend API Error: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

    
@app.route('/classify_event', methods=['POST'])
def classify_event():
    image_files = request.files.getlist('images')
    categories = request.form.getlist('categories')

    if len(image_files) != len(categories):
        return jsonify({'error': 'Mismatch between images and categories'}), 400

    image_file_list_with_categories = list(zip(categories, image_files))

    saved_filenames = []

    for category, file in image_file_list_with_categories:
        ext = file.filename.rsplit('.', 1)[-1].lower()
        safe_category = category.replace("/", "-")  # or replace with "_"
        filename = f"{safe_category}_{uuid.uuid4().hex[:8]}.{ext}"

        save_path = os.path.join("uploads", filename)

        file.save(save_path)

        # 🧠 Save to DB
        img_record = UploadedImage(category=category, filename=filename)
        db.session.add(img_record)
        saved_filenames.append((category, filename))

    db.session.commit()

    print("\n✅ Saved images to DB:", saved_filenames)

    try:
        result = predict_event_from_filenames(saved_filenames)
        return jsonify(result)
    except Exception as e:
        print("❌ Internal error:", e)
        return jsonify({'error': str(e)}), 500



@app.route('/save_outfit', methods=['POST'])
def save_outfit():
    data = request.json
    user_id = data.get('user_id')
    event = data.get('event')
    outfit_paths = data.get('outfit')  # e.g., ['/uploads/xxx.jpg', '/uploads/yyy.jpg']

    if not all([user_id, event, outfit_paths]):
        return jsonify({'error': 'Missing data'}), 400

    # 🔵 Normalize path and get image_ids
    image_ids = []
    for path in outfit_paths:
        # Extract filename only (e.g., '595eb00cfb374e0989e512f2894f0213_upload_0.jpg')
        filename = path.split('/')[-1]
        image_record = ImageModel.query.filter(
            ImageModel.image_path.like(f"%{filename}"),
            ImageModel.user_id == int(user_id)
        ).first()
        if image_record:
            image_ids.append(image_record.id)
        else:
            print(f"⚠️ Image not found for {filename}")

    new_saved = Saved(
        user_id=user_id,
        event=event,
        outfit=outfit_paths,
        clothes_ids=image_ids
    )
    db.session.add(new_saved)
    db.session.commit()

    return jsonify({'message': 'Outfit saved successfully!', 'image_ids': image_ids}), 201


@app.route('/remove_outfit_by_id', methods=['POST'])
def remove_outfit_by_id():
    data = request.get_json()
    outfit_id = data.get("id")

    if not outfit_id:
        return jsonify({"error": "Missing outfit ID"}), 400

    outfit = Saved.query.get(outfit_id)
    if not outfit:
        return jsonify({"error": "Outfit not found"}), 404

    try:
        db.session.delete(outfit)
        db.session.commit()
        return jsonify({"message": "Outfit removed successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500




@app.route('/fp_growth_saved', methods=['GET'])
def fp_growth_saved():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400

    saved_outfits = Saved.query.filter_by(user_id=user_id).all()
    transactions = []
    item_details = {}

    for saved in saved_outfits:
        clothing_ids = []
        for image_id in saved.clothes_ids:
            image = ImageModel.query.filter_by(id=image_id, user_id=int(user_id)).first()
            if image:
                clothing_ids.append(image.id)

                # Get just the filename from the image_path (in case it's a full path)
                filename = os.path.basename(image.image_path)
                full_url = f"{API_URL}/uploads/{filename}"  # Update IP if needed

                item_details[image.id] = {
                    "id": image.id,
                    "image_path": full_url
                }

        if clothing_ids:
            transactions.append(clothing_ids)

    if not transactions:
        return jsonify({'error': 'No transactions available for this user'}), 404

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Compute frequent itemsets
    frequent_itemsets = fpgrowth(df, min_support=0.3, use_colnames=True)

    # Compute rules with confidence
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    print("\n=== FP-Growth Computations ===")
    print(f"Total Transactions: {len(transactions)}")
    print(f"Frequent Itemsets found: {len(frequent_itemsets)}")
    print(f"Association Rules generated: {len(rules)}\n")

    for idx, row in rules.iterrows():
        antecedent = row['antecedents']
        consequent = row['consequents']
        support_x = frequent_itemsets[frequent_itemsets['itemsets'] == antecedent]['support'].values[0]
        support_x_y = row['support']
        confidence = row['confidence']
        lift = row['lift']

        print(f"Rule {idx + 1}: {antecedent} => {consequent}")
        print(f"  Support(X ∪ Y) = {support_x_y:.2f}")
        print(f"  Support(X) = {support_x:.2f}")
        print(f"  Confidence = Support(X ∪ Y) / Support(X) = {support_x_y:.2f} / {support_x:.2f} = {confidence:.2f}")
        print(f"  Lift = {lift:.2f}\n")

    # Return to frontend
    result = []
    for _, row in rules.iterrows():
        combined_items = row['antecedents'].union(row['consequents'])
        items = [item_details[i] for i in combined_items if i in item_details]
        result.append({
            'itemsets': items,
            'support': row['support'],
            'confidence': row['confidence'],
            'lift': row['lift']
        })

    return jsonify({'frequent_itemsets': result})


@app.route("/get_saved_outfits", methods=["GET"])
def get_saved_outfits():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400

    saved_outfits = Saved.query.filter_by(user_id=user_id).all()  # Filter saved outfits by user_id
    if not saved_outfits:
        return jsonify({'saved_outfits': []}), 200

    response_data = []
    for saved in saved_outfits:
        decoded_outfit = saved.outfit if isinstance(saved.outfit, list) else json.loads(saved.outfit)
        response_data.append({
            'id': saved.id,
            'event': saved.event,
            'outfit': decoded_outfit,  # Returns outfits specific to user
            'clothes_ids': saved.clothes_ids
        })

    return jsonify({'saved_outfits': response_data}), 200



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

