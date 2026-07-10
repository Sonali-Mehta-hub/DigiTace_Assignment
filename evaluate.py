"""
Evaluation script.
Runs the held-out test split (data/dataset.csv, split == "test") through
the classifier and reports:
  - overall accuracy
  - a confusion matrix
  - per-category precision/recall
  - raw predictions vs actual labels, saved to results/test_predictions.json

Run:
    python evaluate.py
"""
import time 
import csv
import json
import os
from collections import defaultdict
from classifier import classify_reply, LABELS

DATA_PATH = "data/dataset.csv"
OUT_DIR = "results"
OUT_PATH = os.path.join(OUT_DIR, "test_predictions.json")


def load_dataset(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    data = load_dataset(DATA_PATH)
    test_set = [row for row in data if row["split"] == "test"]
    print(f"Loaded {len(test_set)} held-out test examples.\n")

    predictions = []
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[actual][predicted] += 1

    for row in test_set:
        text, actual = row["text"], row["label"]
        time.sleep(2)  # small delay to avoid rate limits
        try:
            result = classify_reply(text)
            predicted = result["intent"]
            confidence = result["confidence"]
        except Exception as e:
            predicted = "ERROR"
            confidence = 0.0
            print(f"  ! Error classifying '{text[:40]}...': {e}")

        is_correct = predicted == actual
        correct += int(is_correct)
        confusion[actual][predicted] += 1

        predictions.append({
            "text": text,
            "actual": actual,
            "predicted": predicted,
            "confidence": confidence,
            "correct": is_correct,
        })

        status = "\u2713" if is_correct else "\u2717"
        print(f"  {status} [{actual:>20} -> {predicted:<20}] {text[:60]}")

    accuracy = correct / len(test_set) if test_set else 0
    print(f"\nOverall accuracy: {correct}/{len(test_set)} = {accuracy:.1%}\n")

    print("Per-category breakdown:")
    for label in LABELS:
        tp = confusion[label][label]
        actual_total = sum(confusion[label].values())
        predicted_total = sum(confusion[a][label] for a in LABELS)
        recall = tp / actual_total if actual_total else 0
        precision = tp / predicted_total if predicted_total else 0
        print(f"  {label:<20} precision={precision:.2f}  recall={recall:.2f}  (n={actual_total})")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_set),
            "predictions": predictions,
            "confusion_matrix": {a: dict(preds) for a, preds in confusion.items()},
        }, f, indent=2)

    print(f"\nFull results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
