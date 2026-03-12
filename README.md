# Gym Vision 🏋️

A computer vision project that uses deep learning to recognize and classify gym equipment from images. Built with TensorFlow/Keras and transfer learning on MobileNetV2.

---

## Overview

Gym Vision trains an image classifier capable of identifying different types of gym equipment. The model leverages **MobileNetV2** pre-trained on ImageNet and fine-tunes it on a custom gym equipment dataset, making it well-suited for real-world use cases such as:

- Automatic equipment identification in gym images
- Inventory tracking systems
- Workout logging apps that detect equipment from a photo

---

## Project Structure

```
Gym_Vision/
├── equipment_recognition.ipynb   # Main notebook: model training & evaluation
├── Test.ipynb                    # Tutorial notebook using Fashion MNIST
└── data/                         # Place your gym equipment images here (see below)
```

### Notebooks

| Notebook | Purpose |
|---|---|
| `equipment_recognition.ipynb` | Primary project — builds, trains, and evaluates the gym equipment classifier |
| `Test.ipynb` | Introductory Keras tutorial using the Fashion MNIST dataset |

---

## Model Architecture

The classifier uses a two-phase **transfer learning** strategy:

1. **Phase 1 — Feature Extraction (10 epochs)**
   - MobileNetV2 base is frozen
   - Only the new classification head is trained

2. **Phase 2 — Fine-tuning (10 epochs)**
   - Top 54 layers of MobileNetV2 are unfrozen
   - Model is retrained at a lower learning rate (1e-5) to adapt ImageNet features to gym equipment

**Full Architecture:**

```
MobileNetV2 (ImageNet weights, input: 224×224×3)
   └── GlobalAveragePooling2D
   └── Dropout (0.2)
   └── Dense (softmax) — number of output classes = number of equipment directories
```

---

## Dataset Setup

The model expects images organized in subdirectories under `data/`, one directory per class:

```
data/
├── barbell/
│   ├── img1.jpg
│   └── ...
├── dumbbell/
│   ├── img1.jpg
│   └── ...
├── kettlebell/
│   └── ...
└── treadmill/
    └── ...
```

- The dataset is automatically split **80% training / 20% validation**
- Images are resized to **224×224 pixels**
- Corrupted or unreadable image files are detected and removed automatically
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`

---

## Installation

```bash
pip install tensorflow numpy matplotlib seaborn jupyter
```

> TensorFlow 2.x includes Keras. No separate Keras installation is needed.

---

## Usage

1. **Prepare your data** — organize gym equipment images into subdirectories under `data/` as shown above.

2. **Launch the main notebook:**

   ```bash
   jupyter notebook equipment_recognition.ipynb
   ```

3. **Run all cells in order.** The notebook will:
   - Load and validate images from the `data/` directory
   - Download MobileNetV2 pre-trained weights (~14 MB)
   - Train the model in two phases (feature extraction + fine-tuning)
   - Plot training/validation accuracy and loss curves
   - Display sample predictions color-coded by correctness

---

## Training Configuration

| Parameter | Value |
|---|---|
| Input image size | 224 × 224 pixels |
| Batch size | 32 |
| Train / validation split | 80% / 20% |
| Phase 1 — learning rate | 1e-4 |
| Phase 1 — epochs | 10 |
| Phase 2 — learning rate | 1e-5 |
| Phase 2 — epochs | 10 |
| Fine-tuning starts at layer | 100 (unfreezes last 54 layers) |

---

## Results & Visualization

After training, the notebook produces:

- **Accuracy / loss curves** — shows both training and validation metrics across all epochs, with the fine-tuning inflection point marked
- **Prediction grid** — sample validation images with predicted class, confidence score, and color-coded correctness (green = correct, red = incorrect)

---

## Dependencies

| Library | Role |
|---|---|
| TensorFlow / Keras | Model building, training, and evaluation |
| NumPy | Array operations and numerical computing |
| Matplotlib | Plotting training curves and prediction grids |
| Seaborn | Statistical visualization |

---

## License

This project is open source. See the repository for details.
