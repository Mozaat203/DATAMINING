# 🧠 Educational Data Mining: Course Review Sentiment Engine

## 📌 Executive Summary
This repository contains a complete, end-to-end Natural Language Processing (NLP) and Data Mining pipeline designed to extract, preprocess, and classify student feedback from raw course reviews. 

Developed as a rigorous exploration of sequential deep learning and imbalanced data handling, this system categorizes unstructured text into three distinct sentiment classes: **Negative, Neutral, and Positive**. Rather than relying on out-of-the-box APIs, the core classification engine is built from scratch in PyTorch, utilizing a **Bidirectional Long Short-Term Memory (BiLSTM)** network to capture the contextual and sequential nuances of human language.

## 🏗️ Architecture & Pipeline Overview

### 1. Data Ingestion & Noise Reduction
Real-world educational feedback is inherently noisy, multi-lingual, and inconsistently formatted. The data engineering phase strictly enforces quality before modeling:
* **Language Isolation:** Utilizes `langdetect` to filter out non-English records, ensuring a consistent vocabulary space for the embedding layer.
* **Dimensionality Reduction (Target):** Maps a subjective 5-point rating scale to a standardized ternary classification manifold (0: Negative, 1: Neutral, 2: Positive).
* **Text Normalization:** Cleans the raw corpus via a custom utility (`process_tweet`), stripping non-alphanumeric noise, lowercasing, and standardizing the text.
* **Vocabulary Optimization:** Constructs a custom token dictionary restricted to the top 20,000 most frequent words. This bounds the embedding matrix size, preventing the "curse of dimensionality" and optimizing GPU VRAM utilization.
* **Sequence Standardization:** Enforces a maximum sequence length of 100 tokens via active truncation and zero-padding (`<PAD>`).

### 2. Exploratory Data Analysis (EDA) & Mood Mapping
Before modeling, the pipeline executes a statistical analysis to derive actionable insights from the raw text.
* **Volumetric Analysis:** Identifies the top 15 most heavily reviewed courses.
* **Sentiment Ratios:** Calculates percentage-based mood breakdowns per course.
* **Visualization:** Deploys `matplotlib` and `seaborn` to render stacked bar charts and heatmaps, programmatically identifying the happiest, angriest, and most neutral courses based on a mathematically sound threshold (Minimum 90 records).

### 3. Algorithmic Rectification of Class Imbalance
A critical challenge in review-based data mining is extreme class imbalance (e.g., heavily skewed toward generic "Positive" reviews, starving the "Negative" and "Neutral" classes). 

Instead of artificially duplicating data via SMOTE or simply relying on Focal Loss, this project implements a dynamic, mathematical sampling strategy at the DataLoader level:
1. Calculates class distribution frequencies.
2. Computes inverse class weights using the formula: W_i = 1 / (N_i + epsilon)
3. Deploys a PyTorch `WeightedRandomSampler` to explicitly oversample minority classes during the training loop. This ensures the model's loss landscape is not dominated by the majority class, allowing it to learn generalizable features for critical negative feedback.

### 4. Deep Learning Architecture: PyTorch BiLSTM
The core predictive engine is the `SentimentBiLSTM` module, designed to understand context that flows in both directions (e.g., catching the nuance in "The material was good, *but* the audio was terrible").

* **Embedding Layer:** Projects discrete integer tokens into a 100-dimensional continuous vector space. It explicitly ignores index 0 (padding) to conserve gradient calculations.
* **Stacked BiLSTM:** A 2-layer Bidirectional LSTM with 256 hidden dimensions. It processes the sequence forwards and backwards simultaneously.
* **Dropout Regularization:** Applies a 50% Dropout rate between LSTM states and before the fully connected layer to aggressively penalize overfitting.
* **Dense Classifier:** Concatenates the final hidden states from both directional passes and projects them into the 3 output logits.

## ⚙️ Training Dynamics & Optimization
* **Optimizer:** Adam Optimizer parameterized with a learning rate of 0.001 and Weight Decay (1e-3) for L2 regularization.
* **Objective Function:** Standard Cross-Entropy Loss (`nn.CrossEntropyLoss`). 
* **Automated Checkpointing:** The training loop continuously monitors validation loss, overwriting the `best-model.pt` file strictly when the model achieves a new historical minimum. This acts as a programmatic early-stopping mechanism to ensure the deployed weights generalize perfectly to unseen data.

## 📊 Evaluation & Metrics
The model is evaluated on a strictly stratified 20% hold-out test set to ensure minority classes are proportionally represented. 
* **Metrics Tracked:** Generates a comprehensive `classification_report` detailing Precision, Recall, and F1-Score for each specific class.
* **Confusion Matrix:** Outputs a Seaborn-rendered heatmap to visually diagnose false-positive and false-negative clustering.

## 🚀 Execution & Inference

### Prerequisites
Ensure you have a CUDA-capable GPU for optimal training speeds.
```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn langdetect transformers
