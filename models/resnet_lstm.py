# import tensorflow as tf
# ResNet50 = tf.keras.applications.ResNet50
# Input = tf.keras.layers.Input
# Dense = tf.keras.layers.Dense
# Dropout = tf.keras.layers.Dropout
# GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
# Model = tf.keras.models.Model


# def build_model(input_shape=(224, 224, 3), num_classes=1):
#     """
#     Build ResNet model for deepfake detection
#     Args:
#         input_shape: Input shape of images (height, width, channels)
#         num_classes: Number of output classes (1 for binary classification)
#     Returns:
#         Compiled Keras model
#     """
#     # Base ResNet model
#     base_model = ResNet50(
#         include_top=False,
#         weights='imagenet',
#         input_shape=input_shape
#     )
    
#     # Freeze early layers (transfer learning)
#     for layer in base_model.layers[:-30]:
#         layer.trainable = False
    
#     # Print trainable layers info
#     trainable_count = sum([layer.trainable for layer in base_model.layers])
#     print(f"Total layers in base model: {len(base_model.layers)}")
#     print(f"Trainable layers: {trainable_count}")
#     print(f"Frozen layers: {len(base_model.layers) - trainable_count}")
    
#     # Build model
#     inputs = Input(shape=input_shape)
    
#     # Apply base model
#     x = base_model(inputs)
    
#     # Add custom layers
#     x = GlobalAveragePooling2D(name='global_pooling')(x)
#     x = Dense(512, activation='relu', name='dense_1')(x)
#     x = Dropout(0.5, name='dropout_1')(x)
#     x = Dense(256, activation='relu', name='dense_2')(x)
#     x = Dropout(0.5, name='dropout_2')(x)
#     outputs = Dense(num_classes, activation='sigmoid', name='output')(x)
    
#     return Model(inputs=inputs, outputs=outputs, name='resnet50_deepfake')


import tensorflow as tf
ResNet50 = tf.keras.applications.ResNet50
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Model = tf.keras.models.Model

def build_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Build ResNet model for deepfake detection with single image input
    Args:
        input_shape: Input shape of images (height, width, channels)
        num_classes: Number of output classes (1 for binary classification)
    Returns:
        Compiled Keras model
    """
    # Base ResNet model
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze early layers (transfer learning)
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable_count = sum([layer.trainable for layer in base_model.layers])
    print(f"Total layers in base model: {len(base_model.layers)}")
    print(f"Trainable layers: {trainable_count}")
    print(f"Frozen layers: {len(base_model.layers) - trainable_count}")
    
    # Input for a single image
    inputs = Input(shape=input_shape)

    x = base_model(inputs)
 
    x = GlobalAveragePooling2D(name='global_pooling')(x)
    x = Dense(512, activation='relu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(256, activation='relu', name='dense_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    outputs = Dense(num_classes, activation='sigmoid', name='output')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='resnet50_deepfake')

# Build the model
model = build_model()
model.summary()
