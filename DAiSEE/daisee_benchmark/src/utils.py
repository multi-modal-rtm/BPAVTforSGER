import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from scipy.special import softmax
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

LABELS = ['Boredom', 'Engagement', 'Confusion', 'Frustration ']

def calculate_classification_metrics(logits, labels):
    """
    Calculates multiclass classification metrics for each emotion.
    Args:
        logits (np.array): Raw model outputs (logits), shape (N, 4, 4).
        labels (np.array): Ground truth labels (0-3), shape (N, 4).
    """
    preds = np.argmax(logits, axis=2) 
    
    metrics = {}

    for i, emotion in enumerate(LABELS):
        emotion_name = emotion.strip().lower()
        emotion_preds = preds[:, i]
        emotion_labels = labels[:, i]
        emotion_logits = logits[:, i, :]
        emotion_probs = softmax(emotion_logits, axis=1)

        try:
            auc_score = roc_auc_score(emotion_labels, emotion_probs, multi_class='ovr', average='macro')
        except ValueError:
            auc_score = 0.0 
        metrics[f'auc_macro_{emotion_name}'] = auc_score

        accuracy = accuracy_score(emotion_labels, emotion_preds)
        f1 = f1_score(emotion_labels, emotion_preds, average='macro', zero_division=0)
        
        metrics[f'accuracy_{emotion_name}'] = accuracy
        metrics[f'f1_macro_{emotion_name}'] = f1

    exact_match_accuracy = np.mean(np.all(preds == labels, axis=1))
    metrics['exact_match_accuracy'] = exact_match_accuracy

    print("\n--- Classification Report per Emotion ---")
    for i, emotion in enumerate(LABELS):
        print(f"\n--- {emotion.strip()} ---")
        try:
            report = classification_report(labels[:, i], preds[:, i], zero_division=0)
            print(report)
        except Exception as e:
            print(f"Could not generate report: {e}")
            
    return metrics