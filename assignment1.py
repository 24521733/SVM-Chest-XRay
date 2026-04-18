import cv2 as cv
import  os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from SVM import SVM

BASE_DIR = "ChestXRay2017/chest_xray"
def load_data(split: str = "train"):
    normal = "NORMAL"
    pneumonia = "PNEUMONIA"

    images = []
    labels = []

    for img_file in os.listdir(os.path.join(BASE_DIR, split, normal)):
        image = cv.imread(os.path.join(BASE_DIR, split, normal, img_file))
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (128, 128), interpolation=cv.INTER_NEAREST).reshape(-1) / 255.0
        images.append(image)
        labels.append(1)

    for img_file in os.listdir(os.path.join(BASE_DIR, split, pneumonia)):
        image = cv.imread(os.path.join(BASE_DIR, split, pneumonia, img_file))
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (128, 128), interpolation=cv.INTER_NEAREST).reshape(-1) / 255.0
        images.append(image)
        labels.append(-1)

    images = np.stack(images, axis=0)
    return {
        "images": images,
        "labels": np.expand_dims(np.array(labels), axis=-1)
    }

train_data = load_data(split="train")
# dev_data = load_data(split="val")
test_data = load_data(split="test")

mean = np.mean(train_data["images"], axis=0)
std = np.std(train_data["images"], axis=0) + 1e-8

train_data["images"] = (train_data["images"] - mean) / std
test_data["images"] = (test_data["images"] - mean) / std

model = SVM(
    C = 1,
    epochs = 1000,
    lr = 1e-4
)

model.fit(train_data["images"], train_data["labels"])
y_scores = model.predict(test_data["images"])
y_hat = np.sign(y_scores).astype(int)

# Vì Pneumonia là lớp quan trọng nhất (label = -1), ta coi -1 là positive
P = precision_score(test_data["labels"], y_hat, pos_label=-1)
R = recall_score(test_data["labels"], y_hat, pos_label=-1)
F1 = f1_score(test_data["labels"], y_hat, pos_label=-1)

print(f"Precision: {P}")
print(f"Recall: {R}")
print(f"F1: {F1}")
