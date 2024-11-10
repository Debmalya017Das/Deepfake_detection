import tensorflow as tf
import os
from utils.data_processing import create_dataset_from_directory
from models.resnet_lstm import build_model
from utils.utils import setup_logging
# //batch was  100 
def train_model(train_dir, val_dir, epochs=1, batch_size=100):
    """
    Train the deepfake detection model
    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        epochs: Number of training epochs
        batch_size: Batch size for training
    Returns:
        Trained model and training history
    """
    # GPU part not working 
    gpus = tf.config.list_physical_devices('/device:GPU:1')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU setup completed successfully")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision policy set to float16")
        except RuntimeError as e:
            print(f"GPU setup failed: {e}")
    else:
        print("No GPU found. Using CPU.")

    # Create datasets
    print("Creating training dataset...")
    train_dataset = create_dataset_from_directory(
        train_dir,
        batch_size=batch_size
    )
    
    print("Creating validation dataset...")
    val_dataset = create_dataset_from_directory(
        val_dir,
        batch_size=batch_size
    )

    print("Building model...")
    model = build_model()

    # saving model after each epoch and stopping training if the accuracy doesnot increase. ( Here trained for 1 epoch only)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/checkpoints/model_{epoch:02d}.keras',
           save_weights_only=False,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        setup_logging('logs/training_logs')
    ]

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save final model
    model.save('models/checkpoints/final_model.h5')
    
    return model, history


