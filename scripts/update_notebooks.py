"""
scripts/update_notebooks.py
──────────────────────────────
Adds Band Stats, EDA, and CNN fine-tuning cells to notebook files (.ipynb).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "land_atlas_baseline.ipynb",
    ROOT / "notebooks" / "land_atlas_baseline.ipynb"
]

new_cells = [
    {
        "cell_type": "markdown",
        "id": "eda_band_stats_header",
        "metadata": {},
        "source": [
            "## 12. Dataset Band Normalization & Exploratory Data Analysis (EDA)\n",
            "\n",
            "Computes exact 10-band dataset-wide statistics across all 8,356 Sentinel-2 patches to refine `_BAND_MEAN` and `_BAND_STD` vectors, and runs EDA to generate class distribution summaries and spectral signature curves."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "eda_band_stats_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compute exact 10-band dataset-wide mean and std vectors\n",
            "!python -m data.compute_band_stats\n",
            "\n",
            "# Run EDA and feature analysis\n",
            "!python -m data.eda"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "cnn_training_header",
        "metadata": {},
        "source": [
            "## 13. Fine-Tuned PyTorch CNN (Spatial CV & LORO Fine-Tuning)\n",
            "\n",
            "Trains the fine-tuned Multi-Spectral Sentinel-2 CNN (EfficientNet-B0 / ResNet-18 with Spectral Channel Attention) across 5 spatial block folds and Leave-One-Region-Out (LORO) splits. Exports the best checkpoint to ONNX (`models/land_cover.onnx`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "cnn_training_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q onnx onnxscript\n",
            "!git pull origin main\n",
            "!python -m training.train --epochs 30"
        ]
    }
]

def main():
    for nb_path in NOTEBOOKS:
        if not nb_path.exists():
            continue
        print(f"Updating notebook: {nb_path}")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        # Check if cells already exist
        existing_ids = {c.get("id") for c in nb.get("cells", [])}
        if "eda_band_stats_header" not in existing_ids:
            nb["cells"].extend(new_cells)
            with open(nb_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=2)
            print(f"Successfully updated {nb_path.name} with Sections 12 & 13!")
        else:
            print(f"{nb_path.name} already contains Sections 12 & 13.")

if __name__ == "__main__":
    main()
