# 🎵 Sound Classification Audio

A deep learning-based audio classification system that analyzes audio files and predicts their corresponding sound/instrument categories.

The project uses audio preprocessing, Mel Spectrogram feature extraction, data augmentation, and MobileNetV2 transfer learning to build the classification model.

## 📌 Project Overview

Sound Classification Audio is an audio classification project designed to automatically identify the category of an input audio file.

The system processes `.wav` audio files, converts them into Mel Spectrogram-based representations, and uses a MobileNetV2-based deep learning model to classify the audio.

The project consists of:

- 🎵 Audio dataset
- 🧹 Audio preprocessing
- 📊 Feature extraction
- 🔄 Audio augmentation
- 🧠 MobileNetV2 transfer learning
- ⚙️ Backend API
- 🌐 Frontend application
- 💾 Trained model
- 📈 Training logs

## 🎯 Objective

The main objective of this project is to develop an automated audio classification system that can analyze an audio recording and predict its corresponding sound or instrument category.

## ✨ Key Features

- 🎵 Supports audio files such as WAV, MP3, OGG, FLAC and M4A
- 🧹 Automatic audio preprocessing
- 📊 Mel Spectrogram feature extraction
- 📈 Delta and Delta-Delta feature extraction
- 🔄 Audio data augmentation
- ⚖️ Class-weight balancing for imbalanced datasets
- 🧠 MobileNetV2 transfer learning
- 🔧 Two-phase model training
- 🛑 Early stopping
- 📉 Learning-rate reduction
- 💾 Automatic model checkpointing
- 📋 Training logs stored as CSV
- 🌐 Frontend and backend integration

# 🔄 Project Workflow

                    ┌──────────────────────┐
                    │    Audio Dataset     │
                    │       (.wav)         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Audio Preprocessing │
                    │                      │
                    │ • Load audio         │
                    │ • Mono conversion    │
                    │ • Resample to 22050Hz│
                    │ • Trim / Pad to 3 sec│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Data Augmentation  │
                    │                      │
                    │ • Time stretching   │
                    │ • Pitch shifting     │
                    │ • Noise injection    │
                    │ • Time shifting      │
                    │ • Gain variation     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Feature Extraction  │
                    │                      │
                    │ • Mel Spectrogram    │
                    │ • Delta              │
                    │ • Delta-Delta        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Normalization   │
                    │                      │
                    │ Mean / Standard Dev. │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Train / Validation   │
                    │       Split          │
                    │       80 / 20        │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │       MobileNetV2 Model         │
              │                                 │
              │    ImageNet Pretrained Base     │
              └────────────────┬────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐       ┌──────────────────┐
       │    Phase 1       │       │     Phase 2      │
       │                  │       │                  │
       │ Base Frozen      │ ───►  │ Fine-Tuning      │
       │ Train Classifier │       │ Top 30 Layers    │
       └──────────────────┘       └────────┬─────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │   Trained Model      │
                                │      models/         │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │    New Audio Input   │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │      Backend API     │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │       Prediction     │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │  Classification Result│
                                │      Frontend        │
                                └──────────────────────┘
