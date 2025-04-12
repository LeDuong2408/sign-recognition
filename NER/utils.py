from NER_config import *
from seqeval.metrics import classification_report, f1_score
import numpy as np

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [ID2LABLES[l] for l, p in zip(label, pred) if l != -100]
        for label, pred in zip(labels, predictions)
    ]
    true_predictions = [
        [ID2LABLES[p] for l, p in zip(label, pred) if l != -100]
        for label, pred in zip(labels, predictions)
    ]
    
    return {
        "f1": f1_score(true_labels, true_predictions),
        "report": classification_report(true_labels, true_predictions)
    }