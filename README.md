# Momenta Audio Deepfake Detection Take-Home Assessment

## Overview
This repository contains my submission for the Momenta Audio Deepfake Detection Take-Home Assessment. It implements a spectrogram-based Convolutional Neural Network (CNN) to detect AI-generated human speech using a subset of the CD-ADD dataset. The focus is on researching promising approaches, implementing a feasible solution within a 2-day timeframe, and documenting the process with analysis.

## Part 1: Research & Selection

### 1. AASIST (Attention-based Anti-Spoofing with Raw Waveform)
- **Innovation**: Utilizes raw waveform input with an attention mechanism to focus on discriminative temporal regions, enhancing spoofing detection without feature engineering.
- **Metrics**: Equal Error Rate (EER) ~2-5% on ASVspoof 2019 dataset.
- **Why Promising**: Offers potential for real-time detection and is effective for analyzing real conversations and AI-generated speech due to its attention focus.
- **Limitations**: Computationally intensive, requiring significant resources, and may struggle with unseen attack types.

### 2. Spectrogram-based CNN
- **Innovation**: Converts audio to 2D Mel-spectrograms and uses CNNs to detect spatial patterns indicative of deepfakes.
- **Metrics**: EER 4.1-6.5% on CD-ADD dataset (e.g., with Wav2Vec2 and Whisper models).
- **Why Promising**: Balances accuracy and computational efficiency, making it viable for near-real-time detection. Works well with diverse AI-generated speech and real conversation analysis.
- **Limitations**: Requires preprocessing (spectrogram generation), and performance depends on dataset diversity.

### 3. End-to-End Waveform-based Model
- **Innovation**: Processes raw audio waveforms directly using deep neural networks (e.g., RawNet2), avoiding manual feature extraction.
- **Metrics**: EER ~5-7% on ASVspoof 2021 dataset.
- **Why Promising**: Ideal for real-time applications with minimal preprocessing and effective for analyzing real conversations.
- **Limitations**: High computational cost and potential overfitting to training data.

**Choice**: The spectrogram-based CNN was selected for implementation due to its moderate complexity, accessibility with existing libraries (e.g., Librosa, TensorFlow), and suitability. This approach contrasts with AASIST’s resource demands and the end-to-end model’s overfitting risks.

## Part 2: Implementation

### Implementation Comparison
- **AASIST**: Relies on raw waveform with attention, requiring more computational power and complex training, making it less feasible.
- **End-to-End Waveform**: Processes audio directly with deep networks, prone to overfitting with small datasets and resource-heavy, unsuitable for quick prototyping.
- **Spectrogram CNN**: Chosen for its use of preprocessed spectrograms with CNNs, leveraging accessible tools and suitable for rapid development with the CD-ADD dataset.

### Code and Dataset
- **Model**: A spectrogram-based CNN with 3 convolutional layers (32, 64, 128 filters), MaxPooling, and Dense layers for binary classification.
- **Dataset**: Subset of CD-ADD dataset (`dataset_TED-LIUM.zip`, 1.1 GB) from https://openxlab.org.cn/datasets/ylaeo/CD-ADD/tree/main, containing real and AI-generated speech.
- **Notebook**: See `momenta_audio_deepfake.ipynb` for the full implementation.

## Part 3: Documentation & Analysis

### Implementation Process
- **Challenges Encountered**:
  - Initial empty `X` list due to unhandled nested directory structure in the dataset.
  - Uncertainty in labeling due to variable presence of audio files (e.g., missing `valle.wav` in some folders).
- **How Addressed**:
  - Implemented `os.walk` for recursive directory traversal to capture all `.wav` files.
  - Assumed `real.wav` as genuine (label 0) and all other files as deepfakes (label 1), validated with dataset structure.
- **Assumptions Made**:
  - All non-`real.wav` files are deepfakes, based on CD-ADD’s design.
  - The 1.1 GB `dataset_ted` subset is representative for a pilot study.

### Analysis
- **Why Selected**:
  - Chosen for its feasibility with the resouces at my disposal, leveraging accessible libraries and moderate complexity compared to AASIST and end-to-end models.
- **How the Model Works**:
  - Converts audio to 128x128 Mel-spectrograms using Librosa, processed by a CNN with convolutional layers to detect deepfake patterns, followed by dense layers and softmax for binary classification (real vs. fake).
- **Performance Results**:
  - Test accuracy: 0.9937, test loss: 0.0188 after 10 epochs.
  - Rapid convergence with no significant overfitting, as shown in the training plot.
- **Observed Strengths**:
  - Achieves high accuracy (99.37%) on the `dataset_ted` subset, efficient with preprocessing.
- **Observed Weaknesses**:
  - Limited by small dataset size (1.1 GB); potential overfitting to seen data.
- **Suggestions for Future Improvements**:
  - Incorporate data augmentation (e.g., time stretching with `librosa.effects.time_stretch`).
  - Use the full CD-ADD dataset (e.g., `dataset_LibriTTS.zip`, 1.1 TB) for greater diversity.
  - Explore a deeper architecture (e.g., ResNet) for enhanced feature extraction.

### Reflection Questions
1. **Most Significant Challenges**:
   - Handling the nested directory structure and inferring labels without an explicit label file.
2. **How Might This Approach Perform in Real-World Conditions vs. Research Datasets?**
   - May underperform in real-world conditions with noise, accents, or new TTS models due to the limited diversity of the 1.1 GB subset.
3. **What Additional Data or Resources Would Improve Performance?**
   - Full CD-ADD dataset, ASVspoof 2021, or real-world conversational audio; a GPU for scaling training.
4. **How Would You Approach Deploying This Model in a Production Environment?**
   - Use TensorFlow Lite for edge deployment, implement real-time spectrogram generation with audio buffering, monitor model drift, and schedule periodic retraining.
