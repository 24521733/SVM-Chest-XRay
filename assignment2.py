import cv2 as cv
import  os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from SVM_sklearn import SVM
from SVM import SVM as SVM1

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

model.fit(train_data["images"], train_data["labels"].ravel())
y_hat = model.predict(test_data["images"])

# Vì Pneumonia là lớp quan trọng nhất (label = -1), ta coi -1 là positive
P = precision_score(test_data["labels"], y_hat, pos_label=-1)
R = recall_score(test_data["labels"], y_hat, pos_label=-1)
F1 = f1_score(test_data["labels"], y_hat, pos_label=-1)

print(f"Precision: {P}")
print(f"Recall: {R}")
print(f"F1: {F1}")

model1 = SVM1(
    C = 1,
    epochs = 1000,
    lr = 1e-4
)

model1.fit(train_data["images"], train_data["labels"])
y_scores = model1.predict(test_data["images"])
y_hat_ = np.sign(y_scores).astype(int)

# Vì Pneumonia là lớp quan trọng nhất (label = -1), ta coi -1 là positive
P1 = precision_score(test_data["labels"], y_hat_, pos_label=-1)
R1 = recall_score(test_data["labels"], y_hat_, pos_label=-1)
F1_1 = f1_score(test_data["labels"], y_hat_, pos_label=-1)

# So sánh
# Cả 2 model được train với cùng điều kiện:
# - C = 1, epochs = 1000, lr = 1e-4
print("\n── So sánh ────────────────────────────")
print(f"{'Metric':<12} {'Assignment 1':>14} {'Assignment 2':>14}")
print("-" * 42)
print(f"{'Precision':<12} {P1:>14.4f} {P:>14.4f}")
print(f"{'Recall':<12} {R1:>14.4f} {R:>14.4f}")
print(f"{'F1':<12} {F1_1:>14.4f} {F1:>14.4f}")
print("\n── Điều kiện so sánh ──────────────────────────")
print(f"  C       = 1")
print(f"  epochs  = 1000")
print(f"  lr      = 1e-4")
print("\n── Nhận xét ───────────────────────────")
if P1 > P:
    print(f"  - Precision: Assignment 1 cao hơn ({P1:.4f} > {P:.4f})")
else:
    print(f"  - Precision: Assignment 2 cao hơn ({P:.4f} > {P1:.4f})")

if R1 > R:
    print(f"  - Recall: Assignment 1 cao hơn ({R1:.4f} > {R:.4f})")
else:
    print(f"  - Recall: Assignment 2 cao hơn ({R:.4f} > {R1:.4f})")

if F1_1 > F1:
    print(f"  - F1: Assignment 1 cao hơn ({F1_1:.4f} > {F1:.4f})")
else:
    print(f"  - F1: Assignment 2 cao hơn ({F1:.4f} > {F1_1:.4f})")