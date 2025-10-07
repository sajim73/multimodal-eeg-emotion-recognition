# Multimodal EEG-Eye Emotion Recognition

This repository implements a Multimodal Attention-Enhanced Transformer (MAET) for emotion recognition using EEG and eye tracking data from the SEED-VII dataset.

## Overview

The project focuses on developing deep learning models that can recognize human emotions from multimodal physiological signals. Our approach combines EEG (electroencephalogram) brain signals with eye tracking data using transformer-based attention mechanisms.

### Key Features

- **Multimodal Architecture**: Combines EEG and eye tracking modalities for robust emotion recognition
- **Attention-Enhanced Transformer**: Uses multi-head self-attention to capture complex temporal and cross-modal dependencies  
- **Domain Adaptation**: Implements gradient reversal for subject-independent emotion recognition
- **Comprehensive Evaluation**: Supports both subject-dependent and cross-subject validation protocols
- **Flexible Training**: Configurable for different modalities (EEG-only, eye-only, or multimodal)

## Repository Structure

```
multimodal-eeg-emotion-recognition/
├── src/
│   ├── models/
│   │   ├── maet_model.py              # Main MAET architecture
│   │   ├── multiview_embedding.py     # Multi-view feature embedding
│   │   └── attention_blocks.py        # Transformer attention modules
│   ├── data/
│   │   ├── seedvii_dataset.py         # SEED-VII dataset loader
│   │   └── data_preprocessing.py      # Data preprocessing utilities
│   ├── training/
│   │   ├── train_multimodal.py        # Training pipeline and experiments
│   │   └── evaluation_utils.py        # Evaluation metrics and visualization
│   └── utils/
│       ├── gradient_reversal.py       # Domain adversarial training utilities
│       └── safe_forward.py            # Safe model forward pass handling
├── experiments/
│   ├── multimodal_experiments.ipynb   # Main experimental notebook
│   └── subject_dependent_analysis.ipynb
├── configs/
│   └── multimodal_config.yaml         # Configuration file
├── results/                           # Experimental results
├── docs/                             # Documentation
├── tests/                            # Unit tests
├── requirements.txt                  # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-eeg-emotion-recognition.git
cd multimodal-eeg-emotion-recognition
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This project uses the SEED-VII dataset for multimodal emotion recognition. The dataset contains:

- **EEG Data**: 62-channel EEG signals with differential entropy features (310-dimensional)
- **Eye Tracking Data**: Eye movement features including gaze coordinates, pupil diameter, and fixation patterns (33-dimensional)
- **Emotions**: 7 emotion categories (Neutral, Sad, Fear, Happy, Disgust, Surprise, Angry)
- **Subjects**: 20 participants
- **Stimuli**: 80 emotional video clips

### Data Preparation

1. Download the SEED-VII dataset from [the official website](http://bcmi.sjtu.edu.cn/~seed/)
2. Extract the dataset to your desired location
3. Update the `data_dir` path in `configs/multimodal_config.yaml`

Expected directory structure:
```
SEED-VII/
├── EEG_features/
│   ├── 1.mat
│   ├── 2.mat
│   └── ...
└── EYE_features/
    ├── 1.mat
    ├── 2.mat
    └── ...
```

## Usage

### Quick Start

1. **Basic Training**:
```python
from src.training.train_multimodal import MultimodalTrainer
from src.models.maet_model import MAET

# Load configuration
config = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'num_epochs': 100,
    'model': {'embed_dim': 32, 'depth': 3, 'num_heads': 4}
}

# Initialize trainer
trainer = MultimodalTrainer(config)

# Train model
results = trainer.train(
    data_dir="./data/SEED-VII",
    modality="multimodal"  # Options: "eeg", "eye", "multimodal"
)
```

2. **Experiment Scripts**:
```bash
# Subject-dependent experiments
python -m src.training.train_multimodal --experiment subject_dependent --modality multimodal

# Cross-subject experiments  
python -m src.training.train_multimodal --experiment cross_subject --modality eeg
```

3. **Jupyter Notebook**:
```bash
jupyter notebook experiments/multimodal_experiments.ipynb
```

### Configuration

Modify `configs/multimodal_config.yaml` to customize:

- **Model architecture**: embedding dimensions, number of layers, attention heads
- **Training parameters**: learning rate, batch size, epochs, optimization settings
- **Data processing**: normalization, augmentation, subset ratios
- **Evaluation**: metrics, cross-validation settings, robustness testing

### Key Experiments

1. **Subject-Dependent Validation**: Train and test on same subjects
2. **Cross-Subject Validation**: Leave-one-subject-out (LOSO) evaluation  
3. **Modality Comparison**: Compare EEG-only, eye-only, and multimodal performance
4. **Ablation Studies**: Analyze contribution of different model components
5. **Domain Adaptation**: Evaluate generalization across subjects

## Model Architecture

### MAET (Multimodal Attention-Enhanced Transformer)

The core model combines:

- **Multi-View Embedding**: Creates diverse representations of input features
- **Cross-Modal Attention**: Captures interactions between EEG and eye tracking modalities
- **Transformer Encoder**: Processes sequential dependencies with self-attention
- **Domain Adversarial Training**: Learns subject-invariant features for generalization

### Key Components

1. **MultiViewEmbedding**: Projects features into multiple views with gating
2. **TransformerBlock**: Multi-head self-attention with feed-forward networks
3. **GradientReversalLayer**: Enables domain-adversarial training
4. **MAET**: Main model orchestrating all components

## Results

### Preliminary Results (50% of dataset)

| Modality | Subject-Dependent | Cross-Subject |
|----------|-------------------|---------------|
| Multimodal | 60.15% ± 20.36% | 18.92% ± 2.05% |
| EEG-only | 80.52% ± 19.68% | - |
| Eye-only | 49.62% ± 4.83% | - |

*Note: These are preliminary results on a subset of data for development purposes.*

### Performance Analysis

- **EEG-only** shows strong subject-dependent performance
- **Multimodal fusion** provides balanced cross-modal learning
- **Cross-subject** generalization remains challenging (consistent with literature)
- **Domain adaptation** helps reduce subject-specific bias

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest tests/
```

3. Code formatting:
```bash
black src/ tests/
flake8 src/ tests/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Multimodal Attention-Enhanced Transformer for EEG-Eye Emotion Recognition},
  author={Your Name and Others},
  journal={Your Journal},
  year={2024}
}
```

## References

1. Zheng, W. L., et al. (2019). "Multimodal emotion recognition using EEG and eye tracking data." *IEEE Transactions on Affective Computing*.

2. Li, Y., et al. (2022). "MAET: Multimodal Attention-Enhanced Transformer for emotion recognition." *Proceedings of ICASSP*.

3. Ganin, Y., & Lempitsky, V. (2015). "Unsupervised domain adaptation by backpropagation." *Proceedings of ICML*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Primary Contact**: [Your Name](mailto:your.email@university.edu)
- **Issues**: Please use the GitHub issue tracker for questions and bug reports
- **Discussions**: Join our [GitHub Discussions](https://github.com/yourusername/multimodal-eeg-emotion-recognition/discussions)

## Acknowledgments

- SEED-VII dataset provided by BCMI Lab, Shanghai Jiao Tong University
- Inspired by domain adversarial training and transformer architectures
- Thanks to the open-source community for various tools and libraries used

---

**Note**: This repository represents ongoing research in multimodal emotion recognition. Results and methods are continuously being improved and validated.
