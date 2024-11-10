import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
load_img = tf.keras.preprocessing.image.load_img
img_to_array = tf.keras.preprocessing.image.img_to_array
preprocess_input = tf.keras.applications.resnet50.preprocess_input


def split_dataset(source_dir, output_dir, train_size=0.7, val_size=0.15):
    """
    Split images into train, validation and test sets
    Args:
        source_dir: Directory containing 'real' and 'fake' folders with images
        output_dir: Directory to save split datasets
        train_size: Proportion of training data
        val_size: Proportion of validation data (test_size is remaining)
    """

    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

    for label in ['real', 'fake']:
        images = [f for f in os.listdir(os.path.join(source_dir, label)) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        train_images, temp_images = train_test_split(
            images, train_size=train_size, random_state=42
        )
        val_ratio = val_size / (1 - train_size)
        val_images, test_images = train_test_split(
            temp_images, train_size=val_ratio, random_state=42
        )

        for split, image_list in [('train', train_images), 
                                ('val', val_images), 
                                ('test', test_images)]:
            for image in tqdm(image_list, desc=f'Copying {label} images to {split}'):
                shutil.copy2(
                    os.path.join(source_dir, label, image),
                    os.path.join(output_dir, split, label, image)
                )

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image
    Args:
        image_path: Path to image file
        target_size: Target size for the image
    Returns:
        Preprocessed image array
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

def create_dataset_from_directory(directory, batch_size=32):
    """
    Create dataset from directory containing 'real' and 'fake' subdirectories
    Args:
        directory: Root directory containing class subdirectories
        batch_size: Batch size for training
    Returns:
        tf.data.Dataset
    """
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='binary',
        class_names=['real', 'fake'],
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=True,
        seed=42
    )

def augment_image(image):
    """
    Apply data augmentation to a single image
    Args:
        image: Input image array
    Returns:
        Augmented image array
    """
    # Random brightness adjustment
    brightness = tf.random.uniform([], 0.8, 1.2)
    image = image * brightness
    
    # Random contrast adjustment
    contrast = tf.random.uniform([], 0.8, 1.2)
    mean = tf.reduce_mean(image)
    image = (image - mean) * contrast + mean
    
    # Random horizontal flip
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
    
    # Ensure pixel values are in valid range
    image = tf.clip_by_value(image, 0, 255)
    return image

def preprocess_video(video_path, output_frames_dir, frame_interval=30):
    """
    Extract and preprocess frames from a video file
    Args:
        video_path: Path to input video file
        output_frames_dir: Directory to save extracted frames
        frame_interval: Number of frames to skip between extractions
    Returns:
        List of paths to extracted frames
    """
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)
    
    frame_paths = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Resize frame to model input size
            frame = cv2.resize(frame, (224, 224))
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save frame
            frame_path = os.path.join(output_frames_dir, 
                                    f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_paths.append(frame_path)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    return frame_paths

