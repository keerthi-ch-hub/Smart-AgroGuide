import os
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================
# PATHS
# =========================
DATASET_PATH = r"C:\Users\USER\Desktop\agriculture chatbot\data\PlantVillage"
TRAIN_PATH = DATASET_PATH + r"\train"
VAL_PATH = DATASET_PATH + r"\val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =========================
# DATA AUGMENTATION
# =========================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
])

# =========================
# LOAD DATA
# =========================
train_ds = image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    VAL_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

# SAVE LABEL ENCODER (CORRECT FORMAT)
joblib.dump(class_names, "disease_label_encoder.pkl")
print("Label encoder saved!")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =========================
# BUILD MODEL
# =========================
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False  # Freeze initially

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# =========================
# CALLBACKS
# =========================
checkpoint = ModelCheckpoint("disease_model.h5", monitor="val_accuracy", save_best_only=True)
early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

# =========================
# TRAIN TOP LAYERS
# =========================
print("Training top layers...")
model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[checkpoint, early_stop])

# =========================
# FINE-TUNE ALL LAYERS
# =========================
print("Fine-tuning entire EfficientNet...")
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[checkpoint, early_stop])

print("Training complete! Final model saved as disease_model.h5")
