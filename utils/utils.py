# import os
# import json
# import tensorflow as tf
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# def setup_logging(log_dir):
#     """Setup TensorBoard logging"""
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     return tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# def save_metrics(metrics_dict, save_path):
#     """Save metrics to JSON file"""
#     with open(save_path, 'w') as f:
#         json.dump(metrics_dict, f, indent=4)

# def calculate_metrics(y_true, y_pred):
#     """Calculate various metrics for model evaluation"""
#     # Convert predictions to binary
#     y_pred_binary = (y_pred > 0.5).astype(int)
    
#     # Calculate metrics
#     precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average='binary')
#     accuracy = accuracy_score(y_true, y_pred_binary)
#     conf_matrix = confusion_matrix(y_true, y_pred_binary)
    
#     metrics = {
#         'accuracy': float(accuracy),
#         'precision': float(precision),
#         'recall': float(recall),
#         'f1_score': float(f1),
#         'confusion_matrix': conf_matrix.tolist()
#     }
    
#     return metrics
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_curve, auc

def setup_logging(log_dir):
    """Setup TensorBoard logging"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)

def save_metrics(metrics_dict, save_path):
    """Save metrics to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

def calculate_metrics(y_true, y_pred):
    """Calculate various metrics for model evaluation"""
    # Convert predictions to binary
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average='binary')
    accuracy = accuracy_score(y_true, y_pred_binary)
    conf_matrix = confusion_matrix(y_true, y_pred_binary)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': conf_matrix.tolist()
    }
    
    return metrics

def plot_training_history(history, save_path=None):
    """
    Plot training history metrics
    Args:
        history: Keras history object
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(conf_matrix, save_path=None):
    """
    Plot confusion matrix
    Args:
        conf_matrix: 2x2 confusion matrix array
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_pred, save_path=None):
    """
    Plot ROC curve
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        save_path: Optional path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()