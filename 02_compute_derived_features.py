#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_compute_derived_features.py

Phase 2 of the prosodic feature pipeline. Takes the raw per-sentence
acoustic features produced by 01_extract_prosodic_features.py and
computes the final metrics used in the analyses:
duration ratio (Tempo Stability), pause intrusions (Pausal Intrusions),
and F0-based descriptive measures.

See codebook.md for the mapping between the column names used here
and the terminology in the paper (e.g., "duration_ratio"
corresponds to "Tempo Stability").

Input:  ./prosodic_features_imitation_RAW.csv
        (output of 01_extract_prosodic_features.py)
Output: ./prosodic_features_imitation_ENHANCED.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore

INPUT = "./prosodic_features_imitation_RAW.csv"
OUTPUT = "./prosodic_features_imitation_ENHANCED.csv"

df = pd.read_csv(INPUT)

# normalized F0 (z-scored)

df["child_F0_z"] = zscore(df["child_F0_mean"], nan_policy="omit")
df["imitation_F0_z"] = zscore(df["imitation_F0_mean"], nan_policy="omit")
df["F0_z_diff"] = df["child_F0_z"] - df["imitation_F0_z"]

# final pitch drop (start vs end)

def compute_final_drop(f0_min, f0_max):
    if np.isnan(f0_min) or np.isnan(f0_max):
        return np.nan
    return f0_max - f0_min

df["child_final_pitch_drop"] = df.apply(
    lambda r: compute_final_drop(r["child_F0_min"], r["child_F0_max"]), axis=1
)
df["imitation_final_pitch_drop"] = df.apply(
    lambda r: compute_final_drop(r["imitation_F0_min"], r["imitation_F0_max"]), axis=1
)
df["final_drop_diff"] = df["child_final_pitch_drop"] - df["imitation_final_pitch_drop"]


# duration ratio (Tempo Stability)
df["duration_ratio"] = df["dur_child_s"] / df["dur_imitation_s"]

# pause intrusions (Pausal Intrusions)

df["pause_intrusions"] = df["child_pause_count"] - df["imitation_pause_count"]
df["pause_total_diff"] = df["child_pause_total_s"] - df["imitation_pause_total_s"]
df["pause_mean_diff"] = df["child_pause_mean_s"] - df["imitation_pause_mean_s"]

# save
df.to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT}")
print(f"Rows: {len(df)}")
