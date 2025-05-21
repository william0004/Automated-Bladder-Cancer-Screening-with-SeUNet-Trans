# Automated Bladder Cancer Screening with Deep Learning Algorithms

This project is an extension of the original [Cedars-Sinai AI Campus](https://github.com/jlevy44/Cedars_AI_Campus_Tutorials) bladder cancer screening project. It builds upon their work by evaluating, optimizing, and analyzing a **hybrid deep learning segmentation model (seUNet-Trans)** for medical image segmentation and diagnostic prediction.

The focus is to deepen the understanding of segmentation techniques in image processing and assess their applicability to real-world bladder cancer screening via **nuclear-to-cytoplasmic (N/C) ratio** estimation.

## 🔍 Project Objective

- Extend the AI Campus bladder cancer screening pipeline.
- Deepen understanding of image segmentation techniques in medical imaging.
- Evaluate and optimize the seUNet-Trans model for cell nucleus and cytoplasm segmentation.
- Compare seUNet-Trans against traditional machine learning and baseline deep learning models.
- Use predicted N/C ratios to perform diagnostic classification of bladder cancer.

---

## 📁 Dataset

### 1. Urothelial Cell Dataset
- Contains ~300 annotated cell images with ground truth nucleus and cytoplasm masks.
- Used for training and evaluating segmentation algorithms.

### 2. Specimen Dataset
- Contains ~25 cells per patient with diagnostic labels (Negative, Atypical, Suspicious, Positive).
- Used for model validation and N/C ratio-based diagnosis.

> 📌 Data source: [Cedars-Sinai AI Campus](https://github.com/jlevy44/Cedars_AI_Campus_Tutorials/tree/main/Project2)

---

## 🛠️ Environment & Dependencies

### Recommended Setup
- Python 3.8+
- Jupyter Notebook / VS Code
- CUDA-enabled GPU (optional for training)

### Install Dependencies

```bash
pip install opencv-python pandas numpy torchvision segmentation_models_pytorch \
matplotlib seaborn scikit-image scipy tqdm Pillow scikit-learn torch
```

---

# License
This project is for educational and research purposes. Dataset used courtesy of Cedars-Sinai AI Campus.

---
