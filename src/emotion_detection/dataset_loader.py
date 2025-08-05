import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

def load_fer2013_dataset(data_dir, target_size=(48, 48), batch_size=32):
    """
    Optimized data loader for FER2013 with proper preprocessing
    """
    # Moderate augmentation - too much can hurt FER2013 performance
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Simple rescaling works better than complex preprocessing
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # No augmentation for validation and test
    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",  # Keep grayscale for FER2013
        class_mode="categorical",
        subset='training',
        shuffle=True
    )
    
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        subset='validation',
        shuffle=False
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen

def compute_class_weights(train_gen):
    """Compute class weights to handle imbalance"""
    # Get class counts
    class_counts = {}
    for i in range(len(train_gen.classes)):
        class_idx = train_gen.classes[i]
        if class_idx in class_counts:
            class_counts[class_idx] += 1
        else:
            class_counts[class_idx] = 1
    
    total_samples = sum(class_counts.values())
    class_weights = {}
    
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (len(class_counts) * count)
    
    return class_weights