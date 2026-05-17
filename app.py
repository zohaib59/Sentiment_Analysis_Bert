# ULTRA FAST SENTIMENT ANALYSIS + MODEL EVALUATION
# Optimized for Colab GPU
# =========================================================

# !pip install -q transformers accelerate scikit-learn pandas torch tqdm

import pandas as pd
import torch
import numpy as np

from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1. LOAD DATASET
# =========================================================

# Change filename if needed
df = pd.read_csv("/content/your_dataset.csv")

# Optional: limit rows
# df = df.head(25000)

# =========================================================
# 2. COLUMN NAMES
# =========================================================

REVIEW_COLUMN = "review"
LABEL_COLUMN = "sentiment"

# Keep only needed columns
df = df[[REVIEW_COLUMN, LABEL_COLUMN]]

# Remove missing values
df = df.dropna(subset=[REVIEW_COLUMN, LABEL_COLUMN])

# Convert to string
df[REVIEW_COLUMN] = df[REVIEW_COLUMN].astype(str)
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)

# Normalize labels
df[LABEL_COLUMN] = (
    df[LABEL_COLUMN]
    .str.upper()
    .str.strip()
)

# =========================================================
# 3. LOAD MODEL
# =========================================================

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available()
    else torch.float32
)

model.to(device)

# Extra speed boost (PyTorch 2)
try:
    model = torch.compile(model)
    print("Torch compile enabled")
except:
    print("Torch compile skipped")

model.eval()

# =========================================================
# 4. FAST BATCH PREDICTION
# =========================================================

texts = df[REVIEW_COLUMN].tolist()

BATCH_SIZE = 256

all_labels = []
all_scores = []
all_negative_probs = []
all_positive_probs = []

with torch.inference_mode():

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):

        batch_texts = texts[i:i+BATCH_SIZE]

        # Batch tokenization
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        # Move to GPU
        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        # Forward pass
        outputs = model(**inputs)

        # Probabilities
        probs = torch.softmax(outputs.logits, dim=1)

        # Move once to CPU
        probs = probs.cpu().numpy()

        # Extract probabilities
        neg_probs = probs[:, 0]
        pos_probs = probs[:, 1]

        # Predictions
        preds = np.where(
            pos_probs >= 0.5,
            "POSITIVE",
            "NEGATIVE"
        )

        # Confidence scores
        conf_scores = probs.max(axis=1)

        # Save
        all_labels.extend(preds)
        all_scores.extend(conf_scores)
        all_negative_probs.extend(neg_probs)
        all_positive_probs.extend(pos_probs)

# =========================================================
# 5. SAVE PREDICTIONS
# =========================================================

df["predicted_sentiment"] = all_labels
df["confidence_score"] = np.round(all_scores, 4)

df["negative_probability"] = np.round(
    all_negative_probs,
    4
)

df["positive_probability"] = np.round(
    all_positive_probs,
    4
)

# =========================================================
# 6. MODEL EVALUATION
# =========================================================

print("\n================ MODEL PERFORMANCE ================\n")

y_true = df[LABEL_COLUMN]
y_pred = df["predicted_sentiment"]

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Precision
precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

# Recall
recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

# F1 Score
f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

# Print metrics
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# =========================================================
# 7. CONFUSION MATRIX
# =========================================================

print("\n================ CONFUSION MATRIX ================\n")

cm = confusion_matrix(y_true, y_pred)

print(cm)

# =========================================================
# 8. DETAILED REPORT
# =========================================================

print("\n================ CLASSIFICATION REPORT ================\n")

print(
    classification_report(
        y_true,
        y_pred
    )
)

# =========================================================
# 9. SAVE FINAL OUTPUT
# =========================================================

output_file = "/content/sentiment_results.csv"

df.to_csv(output_file, index=False)

print(f"\n✅ Results saved to: {output_file}")

# =========================================================
# 10. QUICK SAMPLE OUTPUT
# =========================================================

print("\n================ SAMPLE PREDICTIONS ================\n")

print(
    df[
        [
            REVIEW_COLUMN,
            LABEL_COLUMN,
            "predicted_sentiment",
            "confidence_score"
        ]
    ].head(10)
)


=====================================================================================================================
# =========================================================
# 11. TEST SINGLE UNSEEN REVIEW
# =========================================================

custom_review = """
Chak De India is one of the most love movie every one love
the movie is the gem of a movie the acting skills are next level
and the story plot is outstanding, but i did not like the movie
"""

# Tokenize
inputs = tokenizer(
    custom_review,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=256
)

# Move to GPU
inputs = {
    k: v.to(device)
    for k, v in inputs.items()
}

# Predict
with torch.inference_mode():

    outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    probs = probs.cpu().numpy()[0]

# Probabilities
negative_prob = probs[0]
positive_prob = probs[1]

# Final Prediction
predicted_sentiment = (
    "POSITIVE"
    if positive_prob >= 0.5
    else "NEGATIVE"
)

confidence_score = max(
    negative_prob,
    positive_prob
)

# =========================================================
# SHOW RESULT
# =========================================================

print("\n================ UNSEEN REVIEW RESULT ================\n")

print("Review:\n")
print(custom_review)

print("\nPredicted Sentiment:")
print(predicted_sentiment)

print(f"\nConfidence Score: {confidence_score:.4f}")

print(f"\nNegative Probability: {negative_prob:.4f}")
print(f"Positive Probability: {positive_prob:.4f}")

