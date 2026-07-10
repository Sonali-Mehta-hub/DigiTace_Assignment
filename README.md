# 🔍 Reply-Intent Classifier

Classify a creator's DM/email reply into an outreach intent category — powered by an LLM few-shot prompt via the **Groq API**, not a locally trained model.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20API-F55036)](https://groq.com/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![License](https://img.shields.io/badge/License-Not%20specified-lightgrey)]()

---

## Overview

This project is our submission for the AI/ML Challenge (Option B: Reply Intent Classifier).

The goal of this project is to automatically classify creator replies to outreach messages into one of five predefined intent categories. This helps talent and influencer agencies reduce manual effort by automatically understanding creator responses.

---

## Problem Statement

When creators reply to outreach emails or direct messages, agency teams must manually determine the creator's intent before taking the next action.

Our system classifies every reply into exactly one of the following categories:

- Interested
- Not Interested
- Pricing Query
- Availability Query
- Unclear

The classifier predicts the intent of a new creator reply and provides a confidence score.

---

## Project Structure

```
.
├── data/
│   ├── dataset.csv
│   └── test_predictions.json
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── cli.py
│
├── README.md
├── requirements.txt
└── LICENSE


## Dataset

The dataset consists of manually created realistic creator replies.

Each example contains:

- Reply text
- Intent label

The five intent labels are:

| Label | Description |
|-------|-------------|
| interested | Creator is willing to collaborate |
| not_interested | Creator declines the collaboration |
| pricing_query | Creator asks about payment or budget |
| availability_query | Creator asks about campaign dates or schedule |
| unclear | Reply does not clearly indicate intent |

The dataset is divided into:

- Training Set (80%)
- Test Set (20%)

---

## Model

Our implementation uses:

- TF-IDF Vectorizer
- Multinomial Naive Bayes Classifier

Pipeline:

```
Reply
      ↓
Text Cleaning
      ↓
TF-IDF Vectorization
      ↓
Naive Bayes Classifier
      ↓
Predicted Intent
      ↓
Confidence Score
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Sonali-Mehta-hub/DigiTace_Assignment.git

cd DigiTace_Assignment
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

```bash
python src/train.py
```

This will:

- Load the dataset
- Train the classifier
- Save the trained model
- Save the TF-IDF vectorizer

---

## Running Predictions

Predict a single creator reply:

```bash
python src/predict.py
```

Example:

```
Enter creator reply:

Sounds interesting! Can you share more details?
```

Output:

```
Predicted Intent:
Interested

Confidence:
0.96
```

---

## CLI Demo

Run:

```bash
python src/cli.py
```

Example:

```
Creator Reply:
I'm interested but what's the budget?

Prediction:
Pricing Query

Confidence:
0.91
```

---

## Model Evaluation

Run

```bash
python src/evaluate.py
```

Evaluation reports:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Predictions are also saved in:

```
data/test_predictions.json
```

---

## Example Predictions

| Creator Reply | Prediction |
|---------------|------------|
| Yes, I'd love to collaborate. | Interested |
| Sorry, not interested at the moment. | Not Interested |
| What's your budget? | Pricing Query |
| When is the campaign scheduled? | Availability Query |
| Maybe, let me think about it. | Unclear |

---

## Error Analysis

Some common misclassifications observed:

- Replies containing both interest and pricing questions may be classified as Pricing Query.
- Very short responses such as "Maybe" are difficult to classify and may be labeled as Unclear.
- Ambiguous replies without sufficient context reduce classifier confidence.

Future improvements include:

- Larger dataset
- Transformer-based models (e.g., BERT)
- Better handling of multi-intent replies

---

## Integration into a Production CRM

### Minimum Viable Integration

The classifier can be exposed as a REST API.

Workflow:

```
Creator Reply
        ↓
CRM Backend
        ↓
Intent Classifier API
        ↓
Predicted Intent
        ↓
Stored in Database
        ↓
Shown in CRM Dashboard
```

When a creator sends a reply:

1. The CRM stores the message.
2. The backend sends the text to the classifier.
3. The classifier predicts the intent.
4. The predicted label and confidence score are saved.
5. The agency dashboard displays the result, allowing agents to prioritize responses.

### Future Enhancements

- Real-time event-driven processing using a message queue.
- Fine-tuned transformer models for improved accuracy.
- Support for multilingual creator replies.
- Multi-label classification for replies expressing multiple intents.
- Human feedback loop for continuous model improvement.

---

## Future Work

Given more time, we would:

- Increase dataset size with more diverse creator replies.
- Experiment with BERT and other transformer models.
- Build a web interface using Flask or Streamlit.
- Add confidence calibration.
- Support multilingual intent classification.

---

## Team


## License

This project was developed as part of the DigiTace AI/ML Challenge and is intended for educational and evaluation purposes.
