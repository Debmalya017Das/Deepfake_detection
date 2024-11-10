import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from utils.data_processing import create_dataset_from_directory

def evaluate_model(model_path, test_dir):
    """Evaluate model with detailed metrics and visualizations, and save results to a text file."""
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Create test dataset
    test_dataset = create_dataset_from_directory(
        test_dir, 
        batch_size=32,
    )

    y_pred_prob = model.predict(test_dataset)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = np.concatenate([y for _, y in test_dataset])


    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

    # Save results to a text file
    output_file="evaluation_metrics2.txt"
    with open(output_file, "w") as f:
        f.write("Evaluation Results:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write("Confusion Matrix:\n")
        for row in metrics['confusion_matrix']:
            f.write(f"{row}\n")

    return metrics
