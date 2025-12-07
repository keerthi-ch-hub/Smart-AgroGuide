# utils/train_crop_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA = BASE / "data" / "crops_dataset.csv"
OUT = BASE / "models" / "crop_model.pkl"
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
# expected columns: N,P,K,ph,rainfall,temperature,crop
X = df[['N','P','K','ph','rainfall','temperature']]
y = df['crop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Train accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))
joblib.dump(clf, OUT)
print("Saved model to", OUT)
