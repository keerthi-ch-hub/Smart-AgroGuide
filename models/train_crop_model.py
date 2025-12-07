# train_crop_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# Load your CSV
df = pd.read_csv(r"C:\Users\USER\Desktop\agriculture chatbot\data\crop_dataset.csv")

# Features (humidity also exists)
X = df[['N','P','K','temperature','humidity','ph','rainfall']].values

# Target column is 'label' (NOT 'crop')
y = df['label'].values

# Encode labels_MPDEL.PY
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

# Train model
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Accuracy
acc = clf.score(X_test, y_test)
print("Test accuracy:", acc)

# Save model
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(clf, MODEL_DIR / "crop_model.pkl")
joblib.dump(le, MODEL_DIR / "crop_label_encoder.pkl")

print("Saved models/crop_model.pkl and crop_label_encoder.pkl")
