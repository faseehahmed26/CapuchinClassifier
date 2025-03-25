  # Capuchin Bird Audio Classifier

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4.1-orange)
![Python](https://img.shields.io/badge/Python-3.7-blue)
![Audio](https://img.shields.io/badge/Audio-Classification-brightgreen)

A deep learning application that detects and counts Capuchin bird calls in forest audio recordings, helping wildlife researchers monitor bird populations without manual audio analysis.

## Overview

This project uses convolutional neural networks to identify the unique call of the Capuchin bird in forest audio recordings. The system processes WAV and MP3 files, converts them to spectrograms, and analyzes them using a trained CNN model. The entire pipeline from audio loading to call detection is automated, making it usable in field research settings.

## Features

- **Audio Preprocessing**: Converts raw audio files to 16kHz mono format for consistent analysis
- **Spectrogram Generation**: Transforms audio samples into time-frequency spectrograms using STFT
- **Deep Learning Classification**: CNN model trained to identify Capuchin calls with high precision
- **Continuous Recording Analysis**: Processes complete forest recordings by windowing and sequential analysis
- **Call Counting**: Implements post-processing to identify distinct calls rather than just detection frames

## Model Performance

- **Precision**: 100% on validation data
- **Recall**: 100% on validation data
- **Training Time**: 4 epochs
- **Model Size**: 47M parameters

## Project Structure

- **Data Loading**: Functions to read and preprocess WAV/MP3 files
- **Model Architecture**: CNN with two convolutional layers, max pooling, and dense layers
- **Forest Parsing**: Pipeline for analyzing continuous recordings by windowing
- **Result Export**: CSV generation for further analysis and visualization

## Getting Started

1. Install requirements:
   ```bash
   pip install -r requirements.txt

Prepare your audio data with two folders:

Parsed_Capuchinbird_Clips: Contains positive samples
Parsed_Not_Capuchinbird_Clips: Contains negative samples
Forest Recordings: Contains full-length recordings to analyze


Train the model or use the pre-trained model:
pythonCopy# Load the pre-trained model
from tensorflow.keras.models import load_model
model = load_model('models/bird_class.h5')

Process forest recordings:
pythonCopy# See Jupyter notebook for complete workflow


Technologies Used

TensorFlow/Keras: Deep learning framework
TensorFlow IO: Audio processing tools
Matplotlib: Visualization
NumPy: Numerical processing
Python: Core programming language

Results
The model successfully identifies Capuchin bird calls in forest recordings with high accuracy, enabling researchers to count and monitor bird populations across multiple recordings without manual analysis.
Future Work

Real-time audio processing capabilities
Mobile deployment for field use
Multi-species classification
Geographic distribution mapping

License
This project is licensed under the MIT License - see the LICENSE file for details.
