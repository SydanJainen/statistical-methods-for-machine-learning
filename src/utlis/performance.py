from typing import Tuple

def confusion_matrix(y_pred, y_test) -> Tuple[int, int, int, int]:
    """
    Compute the confusion matrix components: True Positives (tp), True Negatives (tn),
    False Positives (fp), and False Negatives (fn).
    """
    results = [(1, 1) for predicted, actual in zip(y_pred, y_test) if predicted == actual == 1] + \
              [(0, 0) for predicted, actual in zip(y_pred, y_test) if predicted == actual == 0] + \
              [(1, 0) for predicted, actual in zip(y_pred, y_test) if predicted == 1 and actual == 0] + \
              [(0, 1) for predicted, actual in zip(y_pred, y_test) if predicted == 0 and actual == 1]

    tp, tn, fp, fn = map(sum, zip(*results))
    return tp, tn, fp, fn

def accuracy(tp, tn, total_samples) -> float:
    """Compute the accuracy given True Positives (tp), True Negatives (tn), and the total number of samples."""
    return (tp + tn) / total_samples if total_samples != 0 else 0

def precision(tp, fp) -> float:
    """Compute precision given True Positives (tp) and False Positives (fp)."""
    return tp / (tp + fp) if (tp + fp) != 0 else 0

def recall(tp, fn) -> float:
    """Compute recall given True Positives (tp) and False Negatives (fn)."""
    return tp / (tp + fn) if (tp + fn) != 0 else 0

def f1_score(tp, fn, fp) -> float:
    """Compute F1 score given True Positives (tp), False Negatives (fn), and False Positives (fp)."""
    precision_val = precision(tp, fp)
    recall_val = recall(tp, fn)
    return 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) != 0 else 0

def specificity(tn, fp) -> float:
    """Compute specificity given True Negatives (tn) and False Positives (fp)."""
    return tn / (tn + fp) if (tn + fp) != 0 else 0

def false_positive_rate(fp, tn) -> float:
    """Compute False Positive Rate given False Positives (fp) and True Negatives (tn)."""
    return fp / (fp + tn) if (fp + tn) != 0 else 0

def calculate_performances(y_pred, y_test) -> Tuple[float, float, float, float, float]:
    """Calculate various performance metrics and optionally print detailed information."""
    tp, tn, fp, fn = confusion_matrix(y_pred, y_test)
    total_samples = len(y_test)

    acc = accuracy(tp, tn, total_samples)
    f1_s = f1_score(tp, fn, fp)
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = false_positive_rate(fp, tn)
    spec = specificity(tn, fp)

    return acc, f1_s, spec, fpr, tpr
