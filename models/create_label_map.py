# create_label_map.py
import json
from pathlib import Path

DATA_DIR = Path(r"C:\Users\USER\Desktop\agriculture chatbot\data\PlantVillage")
out = {}

folders = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
for i, name in enumerate(folders):
    out[str(i)] = name

(Path("models") / "label_map.json").parent.mkdir(parents=True, exist_ok=True)
with open(Path("models") / "label_map.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("Wrote models/label_map.json with", len(out), "classes")
