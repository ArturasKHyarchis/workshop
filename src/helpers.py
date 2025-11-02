import re
from sklearn.metrics import accuracy_score, classification_report, f1_score


def preprocess_sentiment(sentiment):
    if sentiment == 'positive':
        return 2
    elif sentiment == 'neutral':
        return 1
    else:
        return 0

def decode_label(value):
    if value == 2:
        return 'positive'
    elif value == 1:
        return 'neutral'
    else:
        return 'negative'

def calculate_metrics(true_labels, pred_labels, print_metrics=False):
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    report = classification_report(true_labels, pred_labels)
    if print_metrics:
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print('Classification Report:')
        print(report)
    return accuracy, f1, report