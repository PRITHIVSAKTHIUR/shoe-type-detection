![44.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/_6WAhmO9_W74Sz2AhwytE.png)

# shoe-type-detection

> shoe-type-detection is a vision-language encoder model fine-tuned from `google/siglip2-base-patch16-512` for **multi-class image classification**. It is trained to detect different types of shoes such as **Ballet Flats**, **Boat Shoes**, **Brogues**, **Clogs**, and **Sneakers**. The model uses the `SiglipForImageClassification` architecture.

> \[!note]
> SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features
> [https://arxiv.org/pdf/2502.14786](https://arxiv.org/pdf/2502.14786)

```py
Classification Report:
              precision    recall  f1-score   support

 Ballet Flat     0.8980    0.9465    0.9216      2000
        Boat     0.9333    0.8750    0.9032      2000
      Brogue     0.9313    0.9490    0.9401      2000
        Clog     0.9244    0.8800    0.9016      2000
     Sneaker     0.9137    0.9480    0.9306      2000

    accuracy                         0.9197     10000
   macro avg     0.9202    0.9197    0.9194     10000
weighted avg     0.9202    0.9197    0.9194     10000
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/e5c_wP09atj7GhXoxUnHW.png)

---

## Label Space: 5 Classes

```
Class 0: Ballet Flat  
Class 1: Boat  
Class 2: Brogue  
Class 3: Clog  
Class 4: Sneaker
```

---

## Install Dependencies

```bash
pip install -q transformers torch pillow gradio hf_xet
```

---

## Inference Code

```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/shoe-type-detection"  # Update with actual model name on Hugging Face
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Updated label mapping
id2label = {
    "0": "Ballet Flat",
    "1": "Boat",
    "2": "Brogue",
    "3": "Clog",
    "4": "Sneaker"
}

def classify_image(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=5, label="Shoe Type Classification"),
    title="Shoe Type Detection",
    description="Upload an image of a shoe to classify it as Ballet Flat, Boat, Brogue, Clog, or Sneaker."
)

if __name__ == "__main__":
    iface.launch()
```

---

## Intended Use

`shoe-type-detection` is designed for:

* **E-Commerce Automation** – Automate product tagging and classification in online retail platforms.
* **Footwear Inventory Management** – Efficiently organize and categorize large volumes of shoe images.
* **Retail Intelligence** – Enable AI-powered search and filtering based on shoe types.
* **Smart Surveillance** – Identify and analyze footwear types in surveillance footage for retail analytics.
* **Fashion and Apparel Research** – Analyze trends in shoe types and customer preferences using image datasets.
