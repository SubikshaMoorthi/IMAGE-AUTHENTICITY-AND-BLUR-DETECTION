import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
import os
# This script will be called by train_models.py to get the data generators
from preprocess_data import auth_train_generator, auth_val_generator, det_train_generator, det_val_generator

# Ensure the 'models' directory exists to save the trained models
os.makedirs('models', exist_ok=True)

# --- Build Authenticity Model with Transfer Learning ---
print("Building Authenticity Model with Transfer Learning...")
base_model_auth = VGG16(
    weights='imagenet',
    include_top=False,  # We don't need the classification layers of VGG16
    input_shape=(224, 224, 3)
)

# Freeze the layers of the base model so they are not retrained
for layer in base_model_auth.layers:
    layer.trainable = False

# Add new custom layers on top of the VGG16 base
x = Flatten()(base_model_auth.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Add a dropout layer to prevent overfitting
predictions_auth = Dense(1, activation='sigmoid')(x)

auth_model = Model(inputs=base_model_auth.input, outputs=predictions_auth)

# Compile the model with a lower learning rate for better fine-tuning
auth_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

auth_model.summary()

print("Training Authenticity Model...")
auth_model.fit(
    auth_train_generator,
    epochs=15,  # Increased epochs for better training
    validation_data=auth_val_generator
)

# Save the trained authenticity model
auth_model.save('models/authenticity_model.h5')
print("Authenticity model saved as models/authenticity_model.h5")

# --- Build Blur Detection Model with Transfer Learning ---
print("\nBuilding Blur Detection Model with Transfer Learning...")
base_model_det = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model_det.layers:
    layer.trainable = False

x = Flatten()(base_model_det.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Add a dropout layer to prevent overfitting
predictions_det = Dense(1, activation='sigmoid')(x)

det_model = Model(inputs=base_model_det.input, outputs=predictions_det)

det_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

det_model.summary()

print("Training Blur Detection Model...")
det_model.fit(
    det_train_generator,
    epochs=15,
    validation_data=det_val_generator
)

# Save the trained detection model
det_model.save('models/detection_model.h5')
print("Blur detection model saved as models/detection_model.h5")