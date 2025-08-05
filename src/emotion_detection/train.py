import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (Dense, Dropout, GlobalAveragePooling2D, 
                         BatchNormalization, Conv2D, MaxPooling2D,
                         Flatten, Input, Add)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
from utils.dataset_loader import load_fer2013_dataset
from utils.dataset_loader import compute_class_weights


def create_custom_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    Custom CNN designed specifically for FER2013
    Fixed architecture to prevent dimension issues
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        # Classification head
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_hybrid_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Alternative model using ResNet-like architecture
    """
    inputs = Input(shape=input_shape)
    
    # Convert grayscale to RGB for pretrained models
    x = Conv2D(3, (1, 1), activation='linear')(inputs)
    
    # Use a lighter pretrained model
    from keras.applications import MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(48, 48, 3),
        input_tensor=x
    )
    
    # Freeze most layers initially
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model, base_model

def train_custom_cnn():
    """
    Train custom CNN approach
    """
    print("Loading FER2013 dataset...")
    train_gen, val_gen, test_gen = load_fer2013_dataset(
        "data/fer2013", 
        target_size=(48, 48), 
        batch_size=32
    )
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Number of classes: {train_gen.num_classes}")
    
    # Compute class weights
    class_weights = compute_class_weights(train_gen)
    print("Class weights:", class_weights)
    
    # Create model
    model = create_custom_cnn(input_shape=(48, 48, 1), num_classes=train_gen.num_classes)
    
    # Compile with standard settings
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("Starting training...")
    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Save model
    try:
        model.save("model/custom_cnn_emotion_model.keras")
        print("Custom CNN model saved!")
    except:
        model.save("model/custom_cnn_emotion_model", save_format='tf')
        print("Model saved in TensorFlow format!")
    
    return model, history

def train_hybrid_approach():
    """
    Train hybrid approach with pretrained backbone
    """
    print("Loading FER2013 dataset...")
    train_gen, val_gen, test_gen = load_fer2013_dataset(
        "data/fer2013", 
        target_size=(48, 48), 
        batch_size=32
    )
    
    # Compute class weights
    class_weights = compute_class_weights(train_gen)
    
    # Create hybrid model
    model, base_model = create_hybrid_model(
        input_shape=(48, 48, 1), 
        num_classes=train_gen.num_classes
    )
    
    # Phase 1: Train with frozen backbone
    print("Phase 1: Training with frozen backbone...")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Phase 2: Fine-tune
    print("Phase 2: Fine-tuning...")
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Save model
    try:
        model.save("model/hybrid_emotion_model.keras")
        print("Hybrid model saved!")
    except:
        model.save("model/hybrid_emotion_model", save_format='tf')
        print("Model saved in TensorFlow format!")
    
    return model, history1, history2

def plot_training_history(history, title="Training History"):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_{title.lower().replace(" ", "_")}.png')
    plt.show()

if __name__ == "__main__":
    print("Choose training approach:")
    print("1. Custom CNN (recommended for FER2013)")
    print("2. Hybrid approach with pretrained backbone")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        model, history = train_custom_cnn()
        plot_training_history(history, "Custom CNN")
    elif choice == "2":
        model, hist1, hist2 = train_hybrid_approach()
        plot_training_history(hist1, "Hybrid Phase 1")
        plot_training_history(hist2, "Hybrid Phase 2")
    else:
        print("Running custom CNN by default...")
        model, history = train_custom_cnn()
        plot_training_history(history, "Custom CNN")