import os
import cv2
import torch
from langsam import LangSAM

# Load model
model = LangSAM()

# Folder paths
input_folder = "C:/Users/uthka/Documents/rugved2/gun-dex/nerf_dataset/train/smoker_gun/smoker_gun"       # contains 15 images
output_folder = "C:/Users/uthka/Documents/rugved2/gun-dex/nerf_dataset/train/smoker_gun/smoker_gun"
os.makedirs(output_folder, exist_ok=True)

# Prompt for segmentation
prompt = "nerf gun segmentation with a bounding box for all the images"

# Process only first 15 images
image_files = sorted(os.listdir(input_folder))[:15]

for i, img_name in enumerate(image_files, start=1):
    img_path = os.path.join(input_folder, img_name)
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Segmentation
    masks, boxes, phrases = model.predict(image_rgb, prompt=prompt)

    # Draw mask and box
    for mask in masks:
        color = (0, 255, 0)
        image_bgr[mask] = (image_bgr[mask] * 0.5 + torch.tensor(color) * 0.5).byte()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save output
    save_path = os.path.join(output_folder, f"segmented_{i}.jpg")
    cv2.imwrite(save_path, image_bgr)
    print(f"Saved segmented image {i} to {save_path}")

print("✅ Segmentation complete for 15 images.")
