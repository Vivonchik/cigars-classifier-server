import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os
from google.cloud import storage

# === НАСТРОЙКА СТРАНИЦЫ ===
st.set_page_config(page_title="Cigar Classifier", layout="centered")

# === ПАРАМЕТРЫ ===
MODEL_PATH = 'model.pt'
LABELS_PATH = 'labels.txt'
GCS_BUCKET = 'cigar-dataset'
MODEL_BLOB = 'models/cigar_classifier_resnet18.pt'
LABELS_BLOB = 'models/class_labels.txt'
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === СКАЧИВАЕМ МОДЕЛЬ И LABELS С GCS ===
@st.cache_resource
def download_from_gcs():
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    if not os.path.exists(MODEL_PATH):
        blob = bucket.blob(MODEL_BLOB)
        blob.download_to_filename(MODEL_PATH)

    if not os.path.exists(LABELS_PATH):
        blob = bucket.blob(LABELS_BLOB)
        blob.download_to_filename(LABELS_PATH)

download_from_gcs()

# === ЗАГРУЗКА МОДЕЛИ ===
with open(LABELS_PATH, 'r') as f:
    CLASS_NAMES = [line.strip() for line in f]
NUM_CLASSES = len(CLASS_NAMES)

@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def predict(image):
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    return CLASS_NAMES[pred.item()], conf.item()

# === UI ===
st.title("🧐 Cigar Classifier")
st.write("Загрузите фото сигары, и модель определит её бренд и модель.")

uploaded_file = st.file_uploader("📤 Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    label, confidence = predict(image)
    st.markdown(f"### 🔍 Предсказание: **{label}**")
    st.markdown(f"Уверенность: `{confidence:.2f}`")