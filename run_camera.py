import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import deque
import numpy as np


# Настройки

MODEL_PATH = "age_gender.pt"  # файл модели
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HISTORY_LEN = 15  # количество кадров для сглаживания


# История для сглаживания

AGE_HISTORY = deque(maxlen=HISTORY_LEN)
GENDER_HISTORY = deque(maxlen=HISTORY_LEN)


# Модель

class AgeGenderNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.gender_head = nn.Linear(512, 1)  # logits
        self.age_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        gender_logits = self.gender_head(x)
        age = self.age_head(x)
        return gender_logits, age

model = AgeGenderNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# Transform

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Детектор лиц

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Камера
cap = cv2.VideoCapture(0)
print("Нажми Q для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        inp = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            gender_logits, age_pred = model(inp)
            gender_prob = torch.sigmoid(gender_logits).item()
            age = age_pred.item()

        # --- Сохраняем в историю ---
        AGE_HISTORY.append(age)
        GENDER_HISTORY.append(gender_prob)

        # --- Сглаживание ---
        smooth_age = sum(AGE_HISTORY) / len(AGE_HISTORY)
        smooth_gender_prob = sum(GENDER_HISTORY) / len(GENDER_HISTORY)

        # UTKFace: 0 = Male, 1 = Female
        gender_label = "Female" if smooth_gender_prob > 0.5 else "Male"
        gender_conf = smooth_gender_prob if smooth_gender_prob > 0.5 else 1 - smooth_gender_prob

        label = f"{gender_label}, {int(smooth_age)} (gender conf: {gender_conf:.2f})"

        # --- Рисуем ---
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

    cv2.imshow("Age & Gender Smoothed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()