"""
training/train.py
──────────────────
Trains the CNN on spatial-block folds, tracked in MLflow.

Supports:
- Gradient accumulation (effective batch scaling on limited GPU / CPU)
- Mixed-precision (fp16 on CUDA, ignored on CPU/MPS)
- Best-model checkpointing per fold
- Final ONNX export from the best fold model

Run from repo root:
    python -m training.train [--fold 0] [--loro highland]
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import yaml
from sklearn.metrics import f1_score, confusion_matrix

from training.dataset import PatchDataset
from training.model   import build_model, export_onnx

_ROOT       = Path(__file__).resolve().parents[1]
_CFG        = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())
_RGN        = yaml.safe_load((_ROOT / "configs" / "regions.yaml").read_text())
CLASSES     = _RGN["classes"]
TC          = _CFG["training"]
SPLITS_DIR  = _ROOT / _CFG["paths"]["splits_dir"]
MODEL_DIR   = _ROOT / _CFG["paths"]["model_dir"]
MLFLOW_URI  = _CFG["paths"]["mlflow_uri"]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, weights=None):
    acc = float((y_pred == y_true).mean())
    wacc = float(np.average(y_pred == y_true, weights=weights)) if weights is not None else acc
    mf1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return dict(acc=acc, weighted_acc=wacc, macro_f1=mf1)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_fold(
    fold_id: str,
    train_csv: Path,
    val_csv: Path,
    device: torch.device,
    run_name: str,
):
    train_ds = PatchDataset(train_csv, augment=True)
    val_ds   = PatchDataset(val_csv,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=TC["batch_size"], shuffle=True,
                          num_workers=TC["num_workers"], pin_memory=(device.type == "cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=TC["batch_size"] * 2, shuffle=False,
                          num_workers=TC["num_workers"])

    model = build_model().to(device)
    class_w = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=TC["lr"], weight_decay=TC["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TC["epochs"])
    scaler    = GradScaler(enabled=(TC["mixed_precision"] and device.type == "cuda"))
    ga_steps  = TC["gradient_accumulation"]

    best_wf1, best_ckpt = -1.0, None
    history = []

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params({
            "fold": fold_id,
            "backbone": _CFG["model"]["backbone"],
            "pretrained": _CFG["model"]["pretrained"],
            "epochs": TC["epochs"],
            "lr": TC["lr"],
            "batch_size": TC["batch_size"],
            "ga_steps": ga_steps,
        })

        for epoch in range(TC["epochs"]):
            model.train()
            optimizer.zero_grad()
            running_loss, n_steps = 0.0, 0

            for step, (imgs, labels, _) in enumerate(train_dl):
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast(enabled=(TC["mixed_precision"] and device.type == "cuda")):
                    logits = model(imgs)
                    loss   = criterion(logits, labels) / ga_steps
                scaler.scale(loss).backward()
                if (step + 1) % ga_steps == 0 or (step + 1) == len(train_dl):
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                running_loss += loss.item() * ga_steps
                n_steps += 1

            scheduler.step()
            train_loss = running_loss / n_steps

            # ── Validation ──
            model.eval()
            all_preds, all_labels, all_weights = [], [], []
            with torch.no_grad():
                for imgs, labels, weights in val_dl:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    preds  = logits.argmax(dim=1).cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(labels.numpy())
                    all_weights.append(weights.numpy())

            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_labels)
            w_val  = np.concatenate(all_weights)
            mets   = compute_metrics(y_true, y_pred, w_val)
            mets["train_loss"] = train_loss
            history.append(mets)

            mlflow.log_metrics({f"val/{k}": v for k, v in mets.items()}, step=epoch)
            print(f"  epoch {epoch+1:03d}  loss {train_loss:.4f}  "
                  f"wAcc {mets['weighted_acc']:.4f}  mF1 {mets['macro_f1']:.4f}")

            if mets["macro_f1"] > best_wf1:
                best_wf1  = mets["macro_f1"]
                best_ckpt = MODEL_DIR / f"ckpt_fold_{fold_id}.pt"
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state": model.state_dict(), "metrics": mets}, best_ckpt)

        mlflow.log_metric("best_macro_f1", best_wf1)
        mlflow.log_artifact(str(best_ckpt))
        print(f"  Best macro-F1: {best_wf1:.4f}  → {best_ckpt}")
    return best_ckpt, best_wf1, history


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None,
                        help="Single fold index (0..N_FOLDS-1). Default: all folds.")
    parser.add_argument("--loro", default=None,
                        help="Leave-one-region-out region name. Overrides --fold.")
    args = parser.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available()
               else torch.device("mps")  if torch.backends.mps.is_available()
               else torch.device("cpu"))
    print(f"Device: {device}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("land_cover_cnn")

    if args.loro:
        region = args.loro
        pairs  = [(f"loro_{region}",
                   SPLITS_DIR / f"loro_{region}_train.csv",
                   SPLITS_DIR / f"loro_{region}_val.csv")]
    elif args.fold is not None:
        pairs = [(str(args.fold),
                  SPLITS_DIR / f"fold_{args.fold}_train.csv",
                  SPLITS_DIR / f"fold_{args.fold}_val.csv")]
    else:
        n = TC["n_folds"]
        pairs = [(str(k),
                  SPLITS_DIR / f"fold_{k}_train.csv",
                  SPLITS_DIR / f"fold_{k}_val.csv")
                 for k in range(n)]

    # Auto-build splits if any required split CSV is missing
    missing = [tr for _, tr, va in pairs if not tr.exists() or not va.exists()]
    if missing:
        print("Split files not found. Automatically building spatial block splits...")
        from data.pipeline.spatial_splits import build_splits
        build_splits()

    results = []
    for fold_id, tr_csv, va_csv in pairs:
        print(f"\n{'='*50}\nFold {fold_id}")
        ckpt, f1, hist = train_fold(fold_id, tr_csv, va_csv, device,
                                     run_name=f"fold_{fold_id}")
        results.append(dict(fold=fold_id, best_macro_f1=f1, ckpt=str(ckpt)))

    # Export best fold to ONNX
    best = max(results, key=lambda r: r["best_macro_f1"])
    model = build_model()
    state = torch.load(best["ckpt"], map_location="cpu")
    model.load_state_dict(state["model_state"])
    export_onnx(model, MODEL_DIR / "land_cover.onnx",
                patch_size=TC["patch_size"], in_channels=_CFG["model"]["in_channels"])

    print("\n===== Summary =====")
    for r in results:
        print(f"  Fold {r['fold']}: macro-F1 = {r['best_macro_f1']:.4f}")
    print(f"\nBest fold: {best['fold']}  ({best['best_macro_f1']:.4f})")

    with open(MODEL_DIR / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
