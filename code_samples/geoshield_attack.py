import os
import torch
import clip
import ollama
import base64
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
input_path   = os.path.join(script_dir, "test_image.jpg")
output_path  = os.path.join(script_dir, "protected_image.jpg")

# ── Config ─────────────────────────────────────────────────────────────────
EPSILON    = 8  / 255   # max perturbation budget (invisible to human eye)
ALPHA      = 1  / 255   # step size per iteration
NUM_STEPS  = 200        # number of PGD iterations
GEO_WEIGHT = 1.0        # weight for suppressing geographic features
NON_GEO_WEIGHT = 0.5    # weight for preserving non-geographic features

# ── Load CLIP ──────────────────────────────────────────────────────────────
print("Loading CLIP model...")
device = "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
print("CLIP loaded.")

# ── Image loading helpers ──────────────────────────────────────────────────
def load_image_tensor(path):
    """Load image as normalized tensor for CLIP"""
    img = Image.open(path).convert("RGB")
    return clip_preprocess(img).unsqueeze(0).to(device)

def load_image_pil(path):
    """Load image as PIL for saving"""
    return Image.open(path).convert("RGB")

def tensor_to_pil(tensor):
    """Convert CLIP-normalized tensor back to PIL image"""
    # Denormalize from CLIP's normalization
    mean = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)
    img  = tensor.squeeze(0).cpu() * std + mean
    img  = torch.clamp(img, 0, 1)
    return TF.to_pil_image(img)

def image_to_base64(path):
    """Convert image file to base64 for Ollama"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ── Step 1: Query LLaVA for original location ─────────────────────────────
def query_llava(image_path, label=""):
    print(f"\nQuerying LLaVA {label}...")
    image_b64 = image_to_base64(image_path)
    response  = ollama.chat(
        model="llava:13b",
        messages=[{
            "role": "user",
            "content": """You are a geolocation expert. Analyze every detail in this image.

Complete this template exactly:

COUNTRY: [your answer]
CITY: [your answer]
LATITUDE: [number only]
LONGITUDE: [number only]
CONFIDENCE: [percentage]
CLUE 1: [specific visual detail]
CLUE 2: [specific visual detail]
CLUE 3: [specific visual detail]

Fill every field. Give coordinates even if uncertain.""",
            "images": [image_b64]
        }]
    )
    result = response["message"]["content"]
    print(result)
    return result

# ── Step 2: Get non-geographic description from LLaVA ─────────────────────
def get_non_geo_description(image_path):
    print("\nGetting non-geographic description...")
    image_b64 = image_to_base64(image_path)
    response  = ollama.chat(
        model="llava:13b",
        messages=[{
            "role": "user",
            "content": """Describe this image in detail but DO NOT mention:
- Any country, city, or region names
- Any landmarks or monuments  
- Any street signs or place names
- Any flags or national symbols
- Any language-specific text

Only describe: objects, colors, people, weather, architecture style, 
vegetation, vehicles. Keep it factual and location-neutral.""",
            "images": [image_b64]
        }]
    )
    description = response["message"]["content"]
    print(f"Non-geo description: {description[:200]}...")
    return description

# ── Step 3: Encode features with CLIP ─────────────────────────────────────
def get_clip_features(image_tensor):
    with torch.no_grad():
        features = clip_model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features

def get_text_features(text):
    tokens   = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features

# ── Step 4: PGD attack ─────────────────────────────────────────────────────
def pgd_attack(image_tensor, geo_features, non_geo_features):
    print(f"\nRunning PGD attack ({NUM_STEPS} steps)...")

    # Random start within epsilon ball
    delta = torch.empty_like(image_tensor).uniform_(-EPSILON, EPSILON)
    delta = delta.requires_grad_(True)

    for step in range(NUM_STEPS):
        adv_image    = image_tensor + delta
        adv_features = clip_model.encode_image(adv_image)
        adv_features = adv_features / adv_features.norm(dim=-1, keepdim=True)

        # Loss 1 — push away from geographic features
        geo_similarity     = torch.nn.functional.cosine_similarity(
            adv_features, geo_features)

        # Loss 2 — stay close to non-geographic features
        non_geo_similarity = torch.nn.functional.cosine_similarity(
            adv_features, non_geo_features)

        # Total loss — maximize geo distance, minimize non-geo distance
        loss = GEO_WEIGHT * geo_similarity - NON_GEO_WEIGHT * non_geo_similarity

        loss.backward()

        with torch.no_grad():
            # PGD step
            delta_data  = delta - ALPHA * delta.grad.sign()
            # Project back into epsilon ball
            delta_data  = torch.clamp(delta_data, -EPSILON, EPSILON)
            # Keep valid pixel range
            delta_data  = torch.clamp(image_tensor + delta_data,
                                      image_tensor.min(),
                                      image_tensor.max()) - image_tensor
            delta.copy_(delta_data)

        delta.grad.zero_()

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{NUM_STEPS} "
                  f"| geo sim: {geo_similarity.item():.4f} "
                  f"| non-geo sim: {non_geo_similarity.item():.4f}")

    return (image_tensor + delta.detach())

# ── Step 5: Visualize results ──────────────────────────────────────────────
def visualize(original_path, protected_path, orig_response, prot_response):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(Image.open(original_path))
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(Image.open(protected_path))
    axes[1].set_title("Protected Image (GeoShield-inspired)", fontsize=12)
    axes[1].axis("off")

    plt.suptitle("GeoShield Attack — Location Privacy Protection", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(script_dir, "geoshield_comparison.png")
    plt.savefig(save_path)
    plt.show()
    print(f"\nComparison saved to {save_path}")

# ── Main pipeline ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("GeoShield-inspired Geo-Privacy Attack")
    print("=" * 60)

    # 1. Query original location
    print("\n[ STEP 1 ] Querying original image location...")
    orig_response = query_llava(input_path, label="(original)")

    # 2. Get non-geographic description
    print("\n[ STEP 2 ] Getting non-geographic description...")
    non_geo_text  = get_non_geo_description(input_path)

    # 3. Extract CLIP features
    print("\n[ STEP 3 ] Extracting CLIP features...")
    image_tensor      = load_image_tensor(input_path)
    image_features    = get_clip_features(image_tensor)
    non_geo_features  = get_text_features(non_geo_text)

    # Geographic features = full image features - non-geographic features
    geo_features = image_features - non_geo_features
    geo_features = geo_features / geo_features.norm(dim=-1, keepdim=True)
    print(f"  Image features shape:   {image_features.shape}")
    print(f"  Non-geo features shape: {non_geo_features.shape}")
    print(f"  Geo features shape:     {geo_features.shape}")

    # 4. Run PGD attack
    print("\n[ STEP 4 ] Running GeoShield attack...")
    protected_tensor = pgd_attack(image_tensor, geo_features, non_geo_features)

    # 5. Save protected image
    print("\n[ STEP 5 ] Saving protected image...")
    protected_pil = tensor_to_pil(protected_tensor)
    protected_pil.save(output_path)
    print(f"  Saved to {output_path}")

    # 6. Query protected image location
    print("\n[ STEP 6 ] Querying protected image location...")
    prot_response = query_llava(output_path, label="(protected)")

    # 7. Visualize
    print("\n[ STEP 7 ] Generating comparison visualization...")
    visualize(input_path, output_path, orig_response, prot_response)

    # 8. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nORIGINAL IMAGE PREDICTION:")
    print(orig_response)
    print("\nPROTECTED IMAGE PREDICTION:")
    print(prot_response)
    print("\nDone! Check geoshield_comparison.png for visual results.")

if __name__ == "__main__":
    main()