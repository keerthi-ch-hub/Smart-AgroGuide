import os
import joblib
from sklearn.preprocessing import LabelEncoder

# Path to your dataset
DATASET_PATH = "../data"   # ← YOUR dataset folder

# List all disease folders
folders = sorted(os.listdir(DATASET_PATH))

print("Found classes:", folders)

# Create label encoder
encoder = LabelEncoder()
encoder.fit(folders)

# Save the encoder
joblib.dump(encoder, "disease_label_encoder.pkl")

print("Created disease_label_encoder.pkl successfully!")
