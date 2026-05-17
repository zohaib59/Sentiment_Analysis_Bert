# IMDB Sentiment Analysis — Ultra Fast GPU Optimized NLP Pipeline
## Overview

This project is a lightning-fast sentiment analysis pipeline built using Hugging Face Transformers and PyTorch, optimized for Google Colab GPU.

The pipeline performs:

* Ultra-fast batch inference
* GPU acceleration
* FP16 mixed precision inference
* Real-time sentiment prediction
* Confidence score generation
* Probability extraction
* Model evaluation metrics
* Unseen review testing

The model used:

`distilbert-base-uncased-finetuned-sst-2-english`

# Features

✅ 20x–50x Faster Than Standard Pipeline

✅ GPU Optimized

✅ Batch Processing

✅ Mixed Precision FP16

✅ Accuracy / F1 / Precision / Recall

✅ Confusion Matrix

✅ Unseen Review Testing

✅ Production-Ready Architecture

✅ Optimized for Large Datasets

---



# Optimization Techniques Used

* Batched Tokenization
* GPU Tensor Processing
* Torch Inference Mode
* FP16 Mixed Precision
* Torch Compile
* Vectorized Probability Extraction
* Reduced Sequence Length
* Single GPU Transfer Strategy

# Evaluation Metrics

The project automatically computes:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

# Unseen Review Testing

You can manually test completely new reviews.

Example:

```python
custom_review = "Amazing movie with excellent acting"
```

---

# Example Output

| Review        | Actual   | Predicted | Confidence |
| ------------- | -------- | --------- | ---------- |
| Amazing movie | POSITIVE | POSITIVE  | 0.9987     |
| Worst acting  | NEGATIVE | NEGATIVE  | 0.9971     |

---

# Tech Stack

* Python
* PyTorch
* Transformers
* Hugging Face
* Scikit-learn
* Pandas
* Google Colab GPU

---

# Future Improvements

* Multilingual Sentiment Analysis
* ONNX Runtime Optimization
* TensorRT Acceleration
* FastAPI Deployment
* Streamlit Web App
* SHAP Explainability
* Real-Time API Inference


#AI #MachineLearning #DeepLearning #NLP #Transformers #PyTorch #HuggingFace #SentimentAnalysis #DataScience #LLM #ArtificialIntelligence #Python #MLOps #GPU #GoogleColab

#AI #NLP #MachineLearning #Transformers #PyTorch
