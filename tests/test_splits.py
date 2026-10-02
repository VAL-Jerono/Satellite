"""
tests/test_splits.py
─────────────────────
Tests for the spatial-block splitting logic (no EE, no files needed).
"""
import numpy as np
import pandas as pd
import pytest


def _make_fake_df(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "lon":    rng.uniform(34.6, 39.9, n),
        "lat":    rng.uniform(-3.9, 2.0,  n),
        "label":  rng.integers(0, 6, n),
        "region": rng.choice(["highland","coast","arid","lake_basin"], n),
        "weight": 1.0,
    })


def test_block_assignment():
    from data.pipeline.spatial_splits import _assign_blocks
    df = _assign_blocks(_make_fake_df())
    assert "block" in df.columns
    assert df["block"].nunique() > 1


def test_buffer_filter_removes_neighbours():
    from data.pipeline.spatial_splits import _assign_blocks, _buffer_filter
    df = _assign_blocks(_make_fake_df(n=500))
    df = df.reset_index(drop=True)

    # Treat first 50 rows as test, rest as train
    test_idx  = np.arange(50)
    train_idx = np.arange(50, len(df))
    kept = _buffer_filter(df, train_idx, test_idx)

    # The buffer should remove at least some training points
    assert len(kept) < len(train_idx), "Buffer filter should remove some points"
    # Kept points should all be in train_idx
    assert set(kept).issubset(set(train_idx))


def test_no_leakage(tmp_path):
    """No training point should be in the same block as a test point."""
    from data.pipeline.spatial_splits import _assign_blocks, _buffer_filter
    df = _assign_blocks(_make_fake_df(n=500))
    df = df.reset_index(drop=True)

    test_idx  = np.arange(50)
    train_idx = np.arange(50, len(df))
    kept = _buffer_filter(df, train_idx, test_idx)

    test_blocks  = set(df.iloc[test_idx]["block"])
    train_blocks = set(df.iloc[kept]["block"])
    overlap      = test_blocks & train_blocks
    assert not overlap, f"Leakage: {overlap}"
