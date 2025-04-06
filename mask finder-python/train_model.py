import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from utils.data_preprocessing import load_data

def create_model(input_shape, num_classes):
    """
    Create a CNN model for face mask detection
    """
    model = Sequential([
        # First Convolutional Layer
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        # Second Convolutional Layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Third Convolutional Layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten and Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Reduce overfitting
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(data_path, model_save_path, img_size=(150, 150), batch_size=32, epochs=20):
    """
    Train the face mask detection model
    """
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Load and preprocess data
    (X_train, y_train), (X_test, y_test), classes = load_data(data_path, img_size)

    # Create model
    model = create_model(input_shape=(img_size[0], img_size[1], 3), num_classes=len(classes))
    print(model.summary())

    # Callbacks
    checkpoint_path = os.path.join(model_save_path, "best_model.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stopping]
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save the entire model
    model.save(os.path.join(model_save_path, "mask_detector_model.h5"))

    # Save class labels
    np.save(os.path.join(model_save_path, "classes.npy"), classes)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, "training_history.png"))

    return model, classes

if __name__ == "__main__":
    data_path = "./data"
    model_save_path = "./models"

    # Train the model
    model, classes = train_model(data_path, model_save_path)
    print(f"Model trained and saved! Classes: {classes}")
