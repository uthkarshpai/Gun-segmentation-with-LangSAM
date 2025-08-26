import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# --- GunDex Info ---
gun_info = {
    "disruptor": {
        "name": "Nerf Elite Disruptor",
        "image": "nerf_images/disruptor.png",
        "range": "27 meters",
        "ammo": "Elite darts",
        "description": "A rapid-fire revolver-style Nerf gun."
    },
    "cycloneshock": {
        "name": "Nerf Mega CycloneShock",
        "image": "nerf_images/cycloneshock.png",
        "range": "23 meters",
        "ammo": "Mega darts",
        "description": "A heavy-duty Nerf blaster with a rotating cylinder."
    }
}

# --- Load Classifier ---
checkpoint = torch.load("nerf_classifier.pth", map_location="cpu")
class_names = checkpoint['class_names']

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Helper to classify a frame ---
def classify_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
        pred_class = output.argmax(1).item()
    return class_names[pred_class]

# --- Helper to create Pokedex card ---
def create_pokedex_card(class_id):
    info = gun_info[class_id]
    card = np.zeros((frame_height, 300, 3), dtype=np.uint8)

    # Gun image
    gun_img = cv2.imread(info["image"])
    if gun_img is not None:
        gun_img = cv2.resize(gun_img, (280, 200))
        card[20:220, 10:290] = gun_img

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 250
    cv2.putText(card, info["name"], (10, y), font, 0.6, (0, 255, 255), 2)
    y += 30
    cv2.putText(card, f"Range: {info['range']}", (10, y), font, 0.5, (255, 255, 255), 1)
    y += 20
    cv2.putText(card, f"Ammo: {info['ammo']}", (10, y), font, 0.5, (255, 255, 255), 1)
    y += 40

    # Wrap description text
    words = info["description"].split()
    line = ""
    for word in words:
        if len(line + word) < 25:
            line += word + " "
        else:
            cv2.putText(card, line, (10, y), font, 0.5, (200, 200, 200), 1)
            line = word + " "
            y += 20
    cv2.putText(card, line, (10, y), font, 0.5, (200, 200, 200), 1)

    return card

# --- Video Processing ---
video_path = "segmented_gun_video.mp4"
cap = cv2.VideoCapture(video_path)

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Classify gun
    class_id = classify_frame(frame)

    # Create Pokedex card
    card = create_pokedex_card(class_id)

    # Combine frame + card horizontally
    combined = np.hstack((frame, card))

    # Show result
    cv2.imshow("GunDex Video", combined)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
