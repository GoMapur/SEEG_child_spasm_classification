
import numpy as np
import pandas as pd
import os

class Meter():
    """Tracks outputs, labels, and losses for a training/validation epoch."""
    def __init__(self):
        self.outputs = []
        self.labels = []
        self.losses = []

    def add(self, output, label, loss):
        self.outputs.append(output)
        self.labels.append(label)
        self.losses.append(loss)

    def accuracy(self):
        outputs = np.concatenate(self.outputs)
        labels = np.concatenate(self.labels)
        preds = (outputs > 0.5).astype(int)
        correct = preds == labels
        return np.mean(correct)

    def loss(self):
        return np.mean(np.concatenate(self.losses))

    def f1(self):
        from sklearn.metrics import f1_score, recall_score, precision_score
        outputs = np.concatenate(self.outputs)
        labels = np.concatenate(self.labels)
        preds = (outputs > 0.5).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        return f1, recall, precision

    def dump_csv(self, filename=None):
        outputs = np.concatenate(self.outputs)
        labels = np.concatenate(self.labels)
        if len(outputs) == 0:
            print("No data available to dump.")
            return None
        res = {
            "outputs": outputs,
            "labels": labels,
        }
        df = pd.DataFrame.from_dict(res)
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename, index=False)
        return df