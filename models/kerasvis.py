from keras.models import load_model
from keras.utils import plot_model

# Load the model
model_path = "checkpoints/final_model.h5"
final_model = load_model(model_path)

# Plot the model structure
plot_model(final_model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)
