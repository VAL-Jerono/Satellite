"""
scripts/export_initial_onnx.py
───────────────────────────────
Exports an initial SpectralCNN ONNX model artifact to models/land_cover.onnx
so that the FastAPI container can start up immediately.
"""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT))

from training.model import build_model, export_onnx

def main():
    out_dir = _ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "land_cover.onnx"
    
    print(f"Building initial PyTorch SpectralCNN model...")
    model = build_model()
    
    print(f"Exporting to ONNX format at {out_file}...")
    export_onnx(model, out_file)
    print("✅ Initial ONNX model artifact successfully created!")

if __name__ == "__main__":
    main()
