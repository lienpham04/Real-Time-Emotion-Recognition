# 🎭 Real-Time Emotion Recognition System

![Project Banner](images/ui_adaptive_demo.png)
<!-- Replace the image above with a collage of your best screenshots -->

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DeBERTa](https://img.shields.io/badge/Model-DeBERTa_V3-yellow)](https://huggingface.co/microsoft/deberta-v3-large)
[![Chrome Extension](https://img.shields.io/badge/Chrome_Extension-Manifest_V3-4285F4?logo=google-chrome&logoColor=white)](https://developer.chrome.com/docs/extensions/)

> **A real-time multi-label emotion recognition system for Zalo and Facebook Messenger using a Hybrid DeBERTa architecture.**

## 📖 Overview

Text-based communication often lacks the non-verbal cues (tone, facial expressions) necessary for true understanding. This project bridges that gap by providing **real-time emotional feedback** directly in your browser.

The system analyzes your messages as you type or receive them, classifying them into **28 distinct emotions** (e.g., *Admiration, Grief, Remorse, Joy*) using a state-of-the-art Hybrid NLP model. It visualizes these emotions via a non-intrusive Chrome Extension.

## ✨ Key Features

*   **🧠 Hybrid Architecture:** Combines **DeBERTa-v3-large** embeddings with **Hand-Engineered Features (HEF)** (13 linguistic dimensions including VADER scores, POS tags).
*   **⚡ Real-Time Analysis:** Uses `MutationObserver` API to detect messages instantly on **Zalo Web** and **Facebook Messenger** without page reloads.
*   **🎨 Adaptive UI:**
    *   **Emoji Mode:** Displays a floating emoji for short greetings (e.g., 👋, 😊).
    *   **Badge Mode:** Displays a text label (e.g., "Sadness 85%") for long, complex sentences.
    *   **Color Coding:** Red 🔴 for negative, Green 🟢 for positive, Blue 🔵 for neutral.
*   **🛡️ VADER Rescue Logic:** A hybrid post-processing mechanism that overrides AI predictions when factual negative sentences (e.g., *"He was injured"*) are incorrectly classified as Neutral.
*   **🌍 Multi-language Support:** Automatically translates Vietnamese inputs to English before inference.

## 🏗️ System Architecture

The system follows a distributed **Client-Server** architecture to offload heavy deep learning computations from the browser.

![System Architecture](images/system_architecture.png)
<!-- Make sure you upload your architecture diagram to an 'images' folder -->

1.  **Frontend (Extension):** Captures DOM text using `MutationObserver` and renders the UI.
2.  **Backend (FastAPI):** Handles Translation -> Tokenization -> Hybrid Inference -> Post-processing.

## 🚀 Installation & Setup

### Prerequisites
*   Python 3.8+
*   Google Chrome (or Chromium-based browser)
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/lienpham04/Real-Time-Emotion-Recognition.git
cd Real-Time-Emotion-Recognition
