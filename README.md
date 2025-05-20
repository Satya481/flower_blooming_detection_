# Flower Blooming Detection using YOLOv8

## Project Overview
This project implements a real-time flower blooming detection system using YOLOv8 (You Only Look Once version 8), a state-of-the-art object detection model. The system can detect and classify flowers into two categories: "Bloomed" and "Unbloomed".

## Table of Contents
1. [Dataset Creation and Annotation](#dataset-creation-and-annotation)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Training Process](#training-process)
5. [Testing and Inference](#testing-and-inference)
6. [Results](#results)
7. [Future Improvements](#future-improvements)
8. [How to Run](#how-to-run)

## Dataset Creation and Annotation

### 1. Data Collection
- Collected images of flowers in both bloomed and unbloomed states
- Images were gathered from various sources and conditions
- Total dataset size: [Number of images] images

### 2. Data Annotation
- Used Roboflow for data annotation
- Annotated images with bounding boxes
- Two classes: "Bloomed" and "Unbloomed"
- Dataset split:
  - Training: 70%
  - Validation: 20%
  - Testing: 10%

### 3. Dataset Structure
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Project Structure
```
flower-blooming-detection/
├── data.yaml
├── train.py
├── test.py
├── requirements.txt
└── runs/
    └── detect/
        └── train/
            └── weights/
                ├── best.pt
                └── last.pt
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd flower-blooming-detection
```

2. Create a virtual environment (recommended):
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Training the Model
```bash
# Activate virtual environment (if not already activated)
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Run training
python train.py
```

Expected output:
```
Ultralytics YOLOv8.0.0  Python-3.8.10 torch-2.0.0 CPU
Training started...
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/20     0G       0.1234     0.5678     0.9012         4        640: 100%|██████████| 56/56 [00:23<00:00, 2.34it/s]
...
```

### 2. Testing the Model
```bash
# Test on a single image
python test.py

# Test on multiple images
python test.py --source test/images/ --conf 0.25
```

### 3. Using the Trained Model
```bash
# For real-time webcam detection
python test.py --source 0

# For video file detection
python test.py --source path/to/video.mp4

# For image directory detection
python test.py --source path/to/images/
```

### 4. Model Export (for deployment)
```bash
# Export to ONNX format
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/train/weights/best.pt'); model.export(format='onnx')"

# Export to TensorRT format
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/train/weights/best.pt'); model.export(format='engine')"
```

## Training Process

### 1. Model Selection
- Used YOLOv8n (nano) as the base model
- Pre-trained on COCO dataset

### 2. Training Configuration
- Epochs: 20
- Batch size: 8
- Image size: 640x640
- Device: CPU
- Confidence threshold: 0.25

### 3. Training Command
```bash
python train.py
```

### 4. Training Parameters
- Learning rate: Default YOLOv8 settings
- Optimizer: Adam
- Loss function: YOLOv8 default loss
- Data augmentation: Default YOLOv8 augmentations

## Testing and Inference

### 1. Testing on Sample Images
```bash
python test.py
```

### 2. Model Performance
- mAP50: [Your mAP score]
- Classes detected: "Bloomed" and "Unbloomed"
- Average inference time: [Your inference time]

## Results
- The model successfully detects and classifies flowers
- Performance metrics:
  - Precision: [Your precision score]
  - Recall: [Your recall score]
  - mAP50: [Your mAP score]

## Future Improvements
1. Increase dataset size
2. Train for more epochs
3. Use GPU for faster training
4. Implement real-time video detection
5. Add more flower species
6. Improve detection accuracy

## Model Usage
The trained model can be used in various applications:
1. Mobile applications
2. Web applications
3. Real-time video processing
4. Automated flower monitoring systems

## Troubleshooting

### Common Issues and Solutions

1. **CUDA/GPU Issues**
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, use CPU
python train.py --device cpu
```

2. **Memory Issues**
```bash
# Reduce batch size
python train.py --batch 4
```

3. **Installation Issues**
```bash
# Update pip
python -m pip install --upgrade pip

# Reinstall requirements
pip install -r requirements.txt --no-cache-dir
```

## License
[Your License]

## Acknowledgments
- YOLOv8 by Ultralytics
- Roboflow for dataset annotation
- [Other acknowledgments]

## Contact
[Your Contact Information] 