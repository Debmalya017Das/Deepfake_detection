import tensorflow as tf
import os
import argparse
from utils.data_processing import split_dataset, create_dataset_from_directory
from models.resnet_lstm import build_model
from utils.utils import plot_training_history, plot_confusion_matrix, plot_roc_curve
from train import train_model
from evaluate import evaluate_model

# couldnot access GPU .
def setup_gpu():
    """Setup GPU for training"""
    gpus = tf.config.list_physical_devices('/device:GPU:1')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU setup completed successfully")
        except RuntimeError as e:
            print(f"GPU setup failed: {e}")
    else:
        print("No GPU found. Using CPU.")

def create_project_structure():
    """Create necessary directories for the project"""
    directories = [
        'data/fake',
        'data/real',
        'datasets/train',
        'datasets/val',
        'datasets/test',
        'models/checkpoints',
        'logs/training_logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def format_metric_value(value):
    """Helper function to format metric values properly"""
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(f'{v:.4f}' if isinstance(v, (int, float)) else str(v) for v in value) + ']'
    elif isinstance(value, (int, float)):
        return f'{value:.4f}'
    else:
        return str(value)

def main(args):
    """Main execution function"""
    try:
        print("Creating project structure...")
        print("Setting up GPU...")
        setup_gpu()
        
    
        test_size = 1.0 - (args.train_size + args.val_size)
        if test_size <= 0:
            raise ValueError("Train and validation sizes must sum to less than 1.0")
        
        print("Step 1: Splitting dataset...")
        split_dataset(
            source_dir=args.data_dir,
            output_dir=args.output_dir,
            train_size=args.train_size,
            val_size=args.val_size
        )
        
        print("Step 2: Training model...")
        train_dir = os.path.join(args.output_dir, 'train')
        val_dir = os.path.join(args.output_dir, 'val')
        
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            raise FileNotFoundError(f"Training or validation directory not found in {args.output_dir}")
        
        model, history = train_model(
            train_dir=train_dir,
            val_dir=val_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print("Step 3: Evaluating model...")
        test_dir = os.path.join(args.output_dir, 'test')
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found in {args.output_dir}")
        
        metrics = evaluate_model(
            model_path='models/checkpoints/final_model.h5',
            test_dir=test_dir
        )
        print("\nFinal Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {format_metric_value(value)}")
        
        print("\nTraining completed successfully!")
        print(f"Results saved in: {args.output_dir}")
        print(f"Model checkpoints saved in: models/checkpoints/")
        print(f"Training logs saved in: logs/training_logs/")
        
    except Exception as e:
        print(f"\nError occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deepfake Detection Training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory containing real and fake images')
    parser.add_argument('--output_dir', type=str, default='datasets',
                      help='Directory to save processed datasets')
    
    # Dataset split arguments
    parser.add_argument('--train_size', type=float, default=0.7,
                      help='Proportion of training data')
    parser.add_argument('--val_size', type=float, default=0.15,
                      help='Proportion of validation data')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    
    # Optional arguments
    parser.add_argument('--model_name', type=str, default='resnet50',
                      help='Model architecture to use')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Input image size')
    
    args = parser.parse_args()
    main(args)

