# Garbage Classification using EfficientNetV2B2 (Week 2 Update 🚀)

This project is a part of the **Shell-Edunet Skills4Future AICTE Internship**, focused on Green Skills using AI technologies. We classify garbage images into 6 categories using a deep learning model trained with TensorFlow and EfficientNetV2B2.

---

## 📌 Problem Statement

To develop an image classification model that can automatically identify the type of garbage from an image. This helps in sorting and waste management using AI-based automation.

---

## 🗂️ Dataset Used

- Source: [Kaggle - TrashType Image Dataset](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)
- Classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- Total Images: 2527
- Train/Val/Test Split: 70% / 20% / 10%

---

## 🧠 Model Architecture

- Base Model: `EfficientNetV2B2` (pretrained on ImageNet)
- Fine-tuned last few layers
- Additional Layers:
  - `GlobalAveragePooling2D`
  - `Dense(128, relu)`
  - `Dropout(0.2)`
  - `Dense(6, softmax)`

---

## 📊 Performance

| Metric      | Value    |
|-------------|----------|
| ✅ Test Accuracy | **96.75%** |
| ✅ Test Loss     | 0.1139 |
| ✅ F1 Score (macro avg) | 0.96 |

> Achieved using transfer learning + data augmentation.

---

## 🧪 Evaluation Report



cardboard 0.96 1.00 0.98 78
glass 0.97 0.99 0.98 96
metal 0.99 1.00 0.99 79
paper 1.00 0.91 0.96 117
plastic 0.97 0.95 0.96 95
trash 0.82 1.00 0.90 27



accuracy                           0.97       492



---

## 🚀 How to Use (Gradio App)

1. Load the model: `garbage_classifier_model.keras`
2. Run the Gradio app to classify images:
```python
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image

model = load_model("garbage_classifier_model.keras")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def classify_image(img):
    img = img.resize((124, 124))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    return f"{class_names[index]} ({prediction[0][index]:.2f} confidence)"

iface = gr.Interface(fn=classify_image, inputs=gr.Image(type='pil'), outputs="text")
iface.launch()


📅 Internship Details
Internship Name: Shell-Edunet Skills4Future AICTE Internship

Duration: 16th June 2025 to 16th July 2025

Focus: Green Skills using AI

Mode: Online with mentorship & hands-on project

📚 Skills Applied
Image classification with TensorFlow & Keras

Transfer learning using EfficientNetV2B2

Data preprocessing and augmentation

Model evaluation and optimization

Gradio app deployment

GitHub version control

📌 Author
Badavath Tharun
AICTE ID: STU682188f9f0bcb1747028217
GitHub: tharun1503
LinkedIn: Tharun’s LinkedIn

🏁 Milestone
✅ Week 1: Dataset processing, baseline model
✅ Week 2: Accuracy improvement (to 96.75%), full evaluation, README update

