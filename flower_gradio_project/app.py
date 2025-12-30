import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import json
from model import FlowerCNN

# -----------------------------
# AYARLAR
# -----------------------------
MODEL_PATH = "flowers_model.pth"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)

# -----------------------------
# SINIF İSİMLERİNİ YÜKLE
# -----------------------------
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# -----------------------------
# MODELİ YÜKLE
# -----------------------------
model = FlowerCNN(num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# GÖRÜNTÜ DÖNÜŞÜMLERİ
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# -----------------------------
# TAHMİN FONKSİYONU
# -----------------------------
def predict_flower(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(probs, 0)

    return {
        class_names[predicted.item()]: float(confidence)
    }

# -----------------------------
# GRADIO ARAYÜZÜ
# -----------------------------
interface = gr.Interface(
    fn=predict_flower,
    inputs=gr.Image(type="pil", label="Çiçek Fotoğrafını Sürükle-Bırak"),
    outputs=gr.Label(num_top_classes=5, label="Tahmin Sonucu"),
    title="Flower Classification with CNN",
    description="Kaggle Flowers Recognition veri seti ile eğitilmiş CNN tabanlı çiçek tanıma sistemi.",
    examples=[
        ["examples/daisy.jpg"],
        ["examples/dandelion.jpg"],
        ["examples/rose.jpg"],
        ["examples/sunflower.jpg"],
        ["examples/tulip.jpg"],
    ]
)

# -----------------------------
# ÇALIŞTIR
# -----------------------------
if __name__ == "__main__":

    interface.launch()
