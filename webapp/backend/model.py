import tensorflow as tf
from tensorflow.keras import models, layers
import os

def get_sr_model(upscale_factor=4, channels=3):
    """
    Reconstructs the SRCNN model architecture.
    Matches the architecture defined in trainning/trainning.ipynb.
    """
    inputs = layers.Input(shape=(None, None, channels))

    # 1. Upsampling using Bilinear Interpolation to match target size
    # Note: notebook said 'bilinear' in code but 'Bicubic' in comment. Code used 'bilinear'.
    x = layers.UpSampling2D(size=(upscale_factor, upscale_factor), interpolation='bilinear')(inputs)

    # 2. Feature Extraction
    x = layers.Conv2D(64, (9, 9), padding='same', activation='relu')(x)

    # 3. Non-linear mapping
    x = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(x)

    # 4. Reconstruction
    outputs = layers.Conv2D(channels, (5, 5), padding='same', activation='linear')(x)

    model = models.Model(inputs, outputs, name="SRCNN")
    return model

def load_model(weights_path):
    """
    Loads the SRCNN model with weights.
    """
    model = get_sr_model()
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Model loaded from {weights_path}")
    else:
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
    return model
