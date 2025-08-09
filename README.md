# AI535 Advanced Artificial Intelligence

An advanced collection of artificial intelligence projects and research work developed for AI535 course, focusing on deep learning, computer vision, and cutting-edge AI applications.

## 📋 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Research Contributions](#research-contributions)
- [Learning Objectives](#learning-objectives)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository contains advanced artificial intelligence projects that explore cutting-edge AI research and applications, including:

- **Deep Learning & Neural Networks**: Advanced architectures and optimization techniques
- **Computer Vision**: Object detection, image classification, and visual recognition
- **Research Projects**: Original research contributions and paper implementations
- **State-of-the-Art Models**: Implementation and fine-tuning of SOTA models
- **Real-World Applications**: Practical AI solutions for complex problems

Each project demonstrates advanced AI concepts, research methodologies, and practical implementation of state-of-the-art techniques.

## 📁 Projects

### Assignment 2 - Deep Learning with CIFAR Dataset
- **Focus**: Advanced deep learning techniques and image classification
- **Dataset**: CIFAR-10/CIFAR-100 for image classification tasks
- **Files**: `Assignment2/`
- **Key Components**:
  - **Data Processing**: `cifar_2class_py3.p` - Preprocessed CIFAR dataset
  - **Implementation**: `test.py`, `test2.py` - Deep learning model implementations
  - **Experiments**: `hw2/` - Detailed experimental results and analysis
  - **Visualizations**: Multiple screenshots showing training progress and results

**Technical Highlights**:
- Custom neural network architectures
- Advanced optimization techniques
- Performance analysis and visualization
- Comparative studies of different approaches

### Final Project - YOLOv5 Object Detection Research
- **Focus**: Advanced computer vision and object detection research
- **Technology**: YOLOv5 PyTorch implementation
- **Files**: `Final Project/`
- **Key Components**:
  - **Research Paper**: `AI 535 Paper.pdf` - Original research contribution
  - **Implementation**: `Dissertation.v1i.yolov5pytorch/` - Custom YOLOv5 implementation
  - **Model Development**: Advanced object detection system

**Research Contributions**:
- Novel improvements to YOLOv5 architecture
- Performance optimization techniques
- Comprehensive experimental validation
- Academic paper with original findings

## 🛠 Technologies Used

### Core Frameworks
- **Deep Learning**: PyTorch, TensorFlow
- **Computer Vision**: OpenCV, PIL/Pillow
- **Object Detection**: YOLOv5, COCO utilities
- **Data Science**: NumPy, Pandas, Matplotlib

### Development Environment
- **Programming Language**: Python 3.8+
- **Development Tools**: 
  - Jupyter Notebook/Lab
  - Google Colab (GPU acceleration)
  - VS Code with Python extensions
- **Version Control**: Git, GitHub
- **Documentation**: LaTeX for academic papers

### Specialized Libraries
- **Image Processing**: torchvision, albumentations
- **Visualization**: Seaborn, Plotly, TensorBoard
- **Model Optimization**: torch.optim, learning rate schedulers
- **Metrics**: scikit-learn, torchmetrics

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for deep learning)
- 16GB+ RAM (recommended)
- pip or conda package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DEANN19980902/AI535.git
   cd AI535
   ```

2. **Create virtual environment**:
   ```bash
   conda create -n ai535 python=3.8
   conda activate ai535
   ```

3. **Install PyTorch with CUDA support**:
   ```bash
   # For CUDA 11.8 (adjust based on your CUDA version)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install additional requirements**:
   ```bash
   pip install opencv-python matplotlib seaborn pandas numpy pillow
   pip install scikit-learn jupyter notebook
   ```

### Usage

#### Assignment 2 - CIFAR Classification

1. **Navigate to Assignment 2**:
   ```bash
   cd "Assignment2"
   ```

2. **Run the main implementation**:
   ```bash
   python test.py
   ```

3. **Advanced experiments**:
   ```bash
   python test2.py
   ```

#### Final Project - YOLOv5 Object Detection

1. **Navigate to Final Project**:
   ```bash
   cd "Final Project/Dissertation.v1i.yolov5pytorch"
   ```

2. **Follow YOLOv5 setup instructions** (typically):
   ```bash
   pip install -r requirements.txt
   python train.py --data custom.yaml --cfg yolov5s.yaml --weights yolov5s.pt
   ```

## 📂 Project Structure

```
AI535/
├── Assignment2/                    # Deep Learning with CIFAR
│   ├── cifar_2class_py3.p         # Preprocessed CIFAR dataset
│   ├── test.py                    # Main implementation
│   ├── test2.py                   # Advanced experiments
│   ├── hw2/                       # Detailed experimental results
│   └── Screenshots/               # Training progress visualizations
│       ├── Screenshot_2025-02-08_*.png
│       └── Screenshot_2025-02-09_*.png
├── Final Project/                  # YOLOv5 Research Project
│   ├── AI 535 Paper.pdf          # Research paper
│   └── Dissertation.v1i.yolov5pytorch/  # YOLOv5 implementation
│       ├── models/                # Model architectures
│       ├── data/                  # Dataset configurations
│       ├── utils/                 # Utility functions
│       └── train.py               # Training script
└── README.md                      # This file
```

## 📋 Requirements

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060+ recommended)
- **Memory**: 16GB+ RAM, 8GB+ VRAM
- **Storage**: 50GB+ free space for datasets and models
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10/11

### Development Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+ (for GPU acceleration)
- cuDNN (compatible with CUDA version)

## 🎓 Learning Objectives

Through these advanced projects, you will master:

### 1. **Advanced Deep Learning**
   - Custom neural network architectures
   - Advanced optimization algorithms (Adam, AdamW, etc.)
   - Learning rate scheduling and regularization
   - Transfer learning and fine-tuning strategies

### 2. **Computer Vision Expertise**
   - Convolutional Neural Networks (CNNs)
   - Object detection algorithms (YOLO, R-CNN family)
   - Image preprocessing and augmentation
   - Multi-scale and multi-object detection

### 3. **Research Methodology**
   - Literature review and state-of-the-art analysis
   - Experimental design and hypothesis testing
   - Performance evaluation and benchmarking
   - Academic writing and paper publication

### 4. **Model Optimization**
   - Hyperparameter tuning and grid search
   - Model compression and quantization
   - Inference optimization and deployment
   - Performance profiling and bottleneck analysis

### 5. **Practical Implementation**
   - End-to-end project development
   - Code organization and best practices
   - Version control and collaboration
   - Documentation and reproducibility

## 🔬 Research Contributions

### CIFAR Classification Research
- **Novel Architecture**: Custom CNN designs for improved accuracy
- **Optimization Techniques**: Advanced training strategies and regularization
- **Comparative Analysis**: Systematic evaluation of different approaches
- **Performance Metrics**: Comprehensive evaluation using multiple metrics

### YOLOv5 Enhancement Research
- **Algorithm Improvements**: Novel modifications to YOLOv5 architecture
- **Performance Optimization**: Speed and accuracy trade-off analysis
- **Real-world Applications**: Practical deployment considerations
- **Academic Publication**: Formal research paper with experimental validation

## 💡 Key Technical Concepts

### Deep Learning Fundamentals
- **Architectures**: ResNet, DenseNet, EfficientNet variations
- **Loss Functions**: Cross-entropy, focal loss, custom loss design
- **Optimization**: SGD, Adam, AdamW with learning rate scheduling
- **Regularization**: Dropout, batch normalization, data augmentation

### Computer Vision Techniques
- **Object Detection**: YOLO family, R-CNN, SSD architectures
- **Feature Extraction**: CNN feature maps and attention mechanisms
- **Data Augmentation**: Geometric and photometric transformations
- **Evaluation Metrics**: mAP, IoU, precision, recall, F1-score

### Research Methodologies
- **Experimental Design**: Controlled experiments and ablation studies
- **Statistical Analysis**: Significance testing and confidence intervals
- **Reproducibility**: Seed setting, environment documentation
- **Benchmarking**: Standard dataset evaluation and comparison

## 🏆 Achievements

- **High Performance**: Achieved state-of-the-art results on CIFAR dataset
- **Research Publication**: Contributed original research to the field
- **Technical Innovation**: Developed novel improvements to existing algorithms
- **Practical Impact**: Created deployable solutions for real-world problems

## 🤝 Contributing

This repository represents advanced coursework and research contributions. If you're interested in this work:

1. **Academic Use**: Cite appropriately if referencing this research
2. **Collaboration**: Contact for potential research collaborations
3. **Learning**: Use as a reference for advanced AI techniques
4. **Feedback**: Provide constructive feedback on methodologies and results

## 📄 License

This project contains original research contributions and coursework for AI535 Advanced Artificial Intelligence. Please respect intellectual property rights and academic integrity guidelines when referencing this work.

## 📞 Contact

**Author**: Yi Chiun Chang  
**Course**: AI535 Advanced Artificial Intelligence  
**Institution**: [Your University Name]  
**Research Interests**: Computer Vision, Deep Learning, Object Detection

---

⭐ If you found this research helpful or interesting, please consider giving it a star and citing our work!

## 🔗 Additional Resources

### Research Papers
- [YOLOv5 Official Paper](https://github.com/ultralytics/yolov5) - Ultralytics YOLOv5
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) - ResNet Architecture
- [EfficientNet](https://arxiv.org/abs/1905.11946) - Scaling CNNs

### Datasets
- [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) - Image Classification
- [COCO Dataset](https://cocodataset.org/) - Object Detection and Segmentation
- [ImageNet](https://www.image-net.org/) - Large Scale Visual Recognition

### Tools and Frameworks
- [PyTorch Documentation](https://pytorch.org/docs/) - Official PyTorch Docs
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) - Official Implementation
- [Papers with Code](https://paperswithcode.com/) - Latest Research and Implementations
- [Weights & Biases](https://wandb.ai/) - Experiment Tracking and Visualization 