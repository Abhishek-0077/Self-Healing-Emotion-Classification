# Self-Healing Emotion Classification (LangGraph + LoRA)

This project implements a robust text classification pipeline using a fine-tuned transformer model and LangGraph. It incorporates a fallback mechanism for low-confidence predictions by prompting the user for clarification.

---

## 📌 Features

- Transformer-based emotion classifier (DistilBERT + LoRA)
- LangGraph DAG to handle:
  - Inference
  - Confidence check
  - Fallback routing
- CLI interaction
- Structured logging (fallbacks, final predictions)

---

## 🚀 How to Run

### 1. Install Requirements

pip install transformers datasets peft langgraph torch


### 2. Run the Classifier
python app.py

