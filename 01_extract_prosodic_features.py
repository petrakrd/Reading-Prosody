#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_extract_prosodic_features.py

Prosodic feature extraction and reading-vs-imitation similarity
(resolution: 10 ms), using praat-parselmouth.

This is Phase 1 of a two-part pipeline:
    01_extract_prosodic_features.py  
        -> reads the reading-imitation audio pairs for each child and
           sentence, extracts prosodic features,
           and computes reading-vs-imitation contour similarity (DTW) on rhythm and melody.
    02_compute_derived_features.py
        -> takes the output of this script and computes the final
           derived metrics reported in the manuscript (duration ratio,
           pause intrusions, etc.).

Each child's own reading recording (Prosodic Reading Task, PRT) is
compared against that same child's own imitation recording (Prosodic
Imitation Task, PIT), NOT against an external adult template. The
"imitation_*" naming below refers to the child's imitation-task
recording, which serves as their individual acoustic baseline.

NOTE ON DATA: raw audio files (.wav) are not included in the OSF
repository. This script and the accompanying alignment table are shared to document  
the feature-extraction procedure; they will not run end-to-end without
the original audio files.

Input:
    ./imitation_baseline_alignment_table.csv
    expected columns:
        - child_id          
        - sentence_id
        - task               -> "reading_vs_imitation"
        - imitation_wav      -> e.g., "child_X_sentence_01.wav"
        - child_wav          -> e.g., "child_X_reading_01.wav"

Output: prosodic_features_imitation_RAW.csv
"""

import os
import math
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import parselmouth
from scipy.stats import pearsonr
from librosa.sequence import dtw
from tqdm import tqdm

# config

MASTER_TABLE = "./imitation_baseline_alignment_table.csv"
OUTPUT_CSV   = "./prosodic_features_imitation_RAW.csv"

# Base directory for building full audio paths.
# Both the reading and imitation recordings are under the same
# per-child folder (see imitation_baseline_alignment_table.csv), so a
# single base directory is enough. Set this to wherever your local
# copy of the (non-public) audio corpus lives.
CHILD_BASE_DIR = "./audio/children"

TARGET_SR    = 16000
TIME_STEP    = 0.01    # 10 ms
N_CONTOUR_PTS = 50     # contour resampling length for similarity
PAUSE_MIN_DUR = 0.15   # minimum pause duration (in sec) - 150 ms
PAUSE_RMS_FRAC = 0.04  # pause threshold as fraction of max RMS

# column names in the dataframe after path-building
IMITATION_COL = "imitation_path"
CHILD_COL = "child_path"
CHILD_ID_COL = "child_id"
SENTENCE_ID_COL = "sentence_id"
TASK_COL = "task"

# original filename columns (in imitation_baseline_alignment_table.csv)
IMITATION_WAV_COL = "imitation_wav"
CHILD_WAV_COL = "child_wav"


# helper functions

def load_audio(path, target_sr=TARGET_SR):
    """
    Load audio:
      1. try soundfile (fast, accurate)
      2. if it fails -> fallback to librosa.load 
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    try:
        y, sr = sf.read(path)
    except Exception as e:
        print(f" soundfile could not read {path}, falling back to librosa ({e})")
        y, sr = librosa.load(path, sr=None, mono=False)

    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)

    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y.astype(np.float32), sr


def extract_pitch_features(y, sr, time_step=TIME_STEP,
                           pitch_floor=75.0, pitch_ceil=600.0,
                           n_contour_pts=N_CONTOUR_PTS):
    """
    F0 extraction (parselmouth-Praat)
    """
    if y is None or len(y) == 0:
        return {
            "F0_mean": math.nan, "F0_sd": math.nan,
            "F0_min": math.nan, "F0_max": math.nan,
            "F0_range": math.nan, "F0_voiced_ratio": 0.0,
            "F0_contour": None
        }

    try:
        snd = parselmouth.Sound(y, sr)
        pitch = snd.to_pitch(time_step=time_step,
                             pitch_floor=pitch_floor,
                             pitch_ceiling=pitch_ceil)
        f0 = pitch.selected_array['frequency']  # Hz, 0 = unvoiced
    except Exception as e:
        print(f" Praat pitch error: {e}")
        return {
            "F0_mean": math.nan, "F0_sd": math.nan,
            "F0_min": math.nan, "F0_max": math.nan,
            "F0_range": math.nan, "F0_voiced_ratio": 0.0,
            "F0_contour": None
        }

    if f0.size == 0:
        return {
            "F0_mean": math.nan, "F0_sd": math.nan,
            "F0_min": math.nan, "F0_max": math.nan,
            "F0_range": math.nan, "F0_voiced_ratio": 0.0,
            "F0_contour": None
        }

    voiced_mask = f0 > 0
    voiced = f0[voiced_mask]
    voiced_ratio = float(np.sum(voiced_mask)) / len(f0)

    if voiced.size == 0:
        stats = {
            "F0_mean": math.nan, "F0_sd": math.nan,
            "F0_min": math.nan, "F0_max": math.nan,
            "F0_range": math.nan, "F0_voiced_ratio": 0.0,
        }
        contour = None
    else:
        stats = {
            "F0_mean": float(np.mean(voiced)),
            "F0_sd": float(np.std(voiced, ddof=1)) if voiced.size > 1 else 0.0,
            "F0_min": float(np.min(voiced)),
            "F0_max": float(np.max(voiced)),
            "F0_range": float(np.max(voiced) - np.min(voiced)),
            "F0_voiced_ratio": voiced_ratio,
        }
        # resample voiced F0 to fixed length
        if voiced.size == 1:
            contour = np.repeat(voiced[0], n_contour_pts)
        else:
            x = np.arange(len(voiced))
            xp = np.linspace(0, len(voiced) - 1, n_contour_pts)
            contour = np.interp(xp, x, voiced)

    stats["F0_contour"] = contour
    return stats


def extract_intensity_features(y, sr, time_step=TIME_STEP, n_contour_pts=N_CONTOUR_PTS):
    """
    Intensity extraction via parselmouth-Praat
    """
    if y is None or len(y) == 0:
        return {"int_mean": math.nan, "int_contour": None}
        
    try:
        snd = parselmouth.Sound(y, sr)
        intensity = snd.to_intensity(time_step=time_step)
        values = intensity.values[0]  # dB
        
        values = values[~np.isnan(values)]
        
        if values.size == 0:
            return {"int_mean": math.nan, "int_contour": None}
            
        int_mean = float(np.mean(values))
        
        # eesample intensity contour to fixed length 
        if values.size == 1:
            contour = np.repeat(values[0], n_contour_pts)
        else:
            x = np.arange(len(values))
            xp = np.linspace(0, len(values) - 1, n_contour_pts)
            contour = np.interp(xp, x, values)
            
        return {"int_mean": int_mean, "int_contour": contour}
        
    except Exception as e:
        print(f" Praat intensity error: {e}")
        return {"int_mean": math.nan, "int_contour": None}


def detect_pauses_rms(y, sr, hop_length=None, min_pause_dur=PAUSE_MIN_DUR, rms_frac=PAUSE_RMS_FRAC):
    """Simple energy-based pause detection"""
    if y is None or len(y) == 0:
        return 0, 0.0, math.nan, math.nan

    if hop_length is None:
        hop_length = int(TIME_STEP * sr)

    total_dur = len(y) / sr

    rms = librosa.feature.rms(y=y, frame_length=2*hop_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    if rms.size == 0:
        return 0, 0.0, math.nan, math.nan

    thr = rms_frac * np.max(rms)
    is_sil = rms < thr

    pauses = []
    in_pause = False
    start_time = None

    for t, sil in zip(times, is_sil):
        if sil and not in_pause:
            in_pause = True
            start_time = t
        elif not sil and in_pause:
            end_time = t
            dur = end_time - start_time
            if dur >= min_pause_dur:
                pauses.append(dur)
            in_pause = False

    if in_pause and start_time is not None:
        end_time = times[-1]
        dur = end_time - start_time
        if dur >= min_pause_dur:
            pauses.append(dur)

    pause_count = len(pauses)
    total_pause = float(np.sum(pauses)) if pauses else 0.0
    mean_pause = float(np.mean(pauses)) if pauses else math.nan
    speech_ratio = (total_dur - total_pause) / total_dur if total_dur > 0 else math.nan

    return pause_count, total_pause, mean_pause, speech_ratio


def contour_similarity(imitation_contour, child_contour):
    """
    Correlation + DTW distance between two contours,
    comparing the child's reading-task contour against that same child's
    imitation-task (baseline) contour.
    """
    if imitation_contour is None or child_contour is None:
        return math.nan, math.nan

    def zscore(x):
        x = np.asarray(x, dtype=np.float32)
        if x.size <= 1 or np.allclose(x, x[0]):
            return np.zeros_like(x)
        return (x - np.mean(x)) / np.std(x, ddof=1)

    a = zscore(imitation_contour)
    c = zscore(child_contour)

    n = min(len(a), len(c))
    if n <= 1:
        return math.nan, math.nan
    a = a[:n]
    c = c[:n]

    try:
        corr, _ = pearsonr(a, c)
    except Exception:
        corr = math.nan

    D, wp = dtw(a.reshape(1, -1), c.reshape(1, -1))
    dtw_dist = float(D[-1, -1] / len(wp[0]))

    return float(corr), dtw_dist


# pipeline

def main():
    if not os.path.isfile(MASTER_TABLE):
        raise FileNotFoundError(f"MASTER_TABLE not found: {MASTER_TABLE}")

    df = pd.read_csv(MASTER_TABLE)

    required_base = [CHILD_ID_COL, SENTENCE_ID_COL, TASK_COL, IMITATION_WAV_COL, CHILD_WAV_COL]
    missing_base = [col for col in required_base if col not in df.columns]
    if missing_base:
        raise ValueError(f"Missing expected columns in imitation_baseline_alignment_table.csv: {missing_base}")

    # path build
    if IMITATION_COL not in df.columns or CHILD_COL not in df.columns:
        def make_imitation_path(r):
            child_id = str(r[CHILD_ID_COL]).zfill(3)
            fname = str(r[IMITATION_WAV_COL])
            return os.path.join(CHILD_BASE_DIR, f"child_{child_id}", "imitation", fname)

        def make_child_path(r):
            child_id = str(r[CHILD_ID_COL]).zfill(3)
            task = str(r[TASK_COL])
            fname = str(r[CHILD_WAV_COL])
            return os.path.join(CHILD_BASE_DIR, f"child_{child_id}", "reading", fname)

        df[IMITATION_COL] = df.apply(make_imitation_path, axis=1)
        df[CHILD_COL] = df.apply(make_child_path, axis=1)

    missing_cols = [col for col in [CHILD_ID_COL, SENTENCE_ID_COL, TASK_COL, IMITATION_COL, CHILD_COL] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns after path-building: {missing_cols}")

    rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing pairs"):
        child_id = row[CHILD_ID_COL]
        sentence_id = row[SENTENCE_ID_COL]
        task = row[TASK_COL]
        imitation_path = str(row[IMITATION_COL])
        child_path = str(row[CHILD_COL])

        if pd.isna(imitation_path) or pd.isna(child_path):
            continue

        try:
            # audio load
            y_imitation, sr_i = load_audio(imitation_path, TARGET_SR)
            y_child, sr_c = load_audio(child_path, TARGET_SR)

            dur_imitation = len(y_imitation) / sr_i
            dur_child = len(y_child) / sr_c

            # pitch
            imitation_f0 = extract_pitch_features(y_imitation, sr_i)
            child_f0 = extract_pitch_features(y_child, sr_c)

            # intensity
            imitation_int = extract_intensity_features(y_imitation, sr_i)
            child_int = extract_intensity_features(y_child, sr_c)

            # pauses
            i_pause_count, i_pause_total, i_pause_mean, i_speech_ratio = detect_pauses_rms(y_imitation, sr_i)
            c_pause_count, c_pause_total, c_pause_mean, c_speech_ratio = detect_pauses_rms(y_child, sr_c)

            dur_imitation_speech = i_speech_ratio * dur_imitation if not math.isnan(i_speech_ratio) else math.nan
            dur_child_speech = c_speech_ratio * dur_child if not math.isnan(c_speech_ratio) else math.nan

            # contour similarity with DTW
            corr_f0, dtw_f0 = contour_similarity(
                imitation_f0["F0_contour"], child_f0["F0_contour"]
            )

            corr_int, dtw_int = contour_similarity(
                imitation_int["int_contour"], child_int["int_contour"]
            )

            out_row = {
                "child_id": child_id,
                "sentence_id": sentence_id,
                "task": task,
                "imitation_path": imitation_path,
                "child_path": child_path,

                "dur_imitation_s": dur_imitation,
                "dur_child_s": dur_child,
                "dur_imitation_speech_s": dur_imitation_speech,
                "dur_child_speech_s": dur_child_speech,

                "imitation_F0_mean": imitation_f0["F0_mean"],
                "imitation_F0_sd": imitation_f0["F0_sd"],
                "imitation_F0_min": imitation_f0["F0_min"],
                "imitation_F0_max": imitation_f0["F0_max"],
                "imitation_F0_range": imitation_f0["F0_range"],
                "imitation_F0_voiced_ratio": imitation_f0["F0_voiced_ratio"],

                "child_F0_mean": child_f0["F0_mean"],
                "child_F0_sd": child_f0["F0_sd"],
                "child_F0_min": child_f0["F0_min"],
                "child_F0_max": child_f0["F0_max"],
                "child_F0_range": child_f0["F0_range"],
                "child_F0_voiced_ratio": child_f0["F0_voiced_ratio"],

                "imitation_int_mean_dB": imitation_int["int_mean"],
                "child_int_mean_dB": child_int["int_mean"],

                "F0_contour_corr": corr_f0,
                "F0_contour_dtw": dtw_f0,

                "int_contour_corr": corr_int,
                "int_contour_dtw": dtw_int,

                "imitation_pause_count": i_pause_count,
                "imitation_pause_total_s": i_pause_total,
                "imitation_pause_mean_s": i_pause_mean,
                "imitation_speech_ratio": i_speech_ratio,

                "child_pause_count": c_pause_count,
                "child_pause_total_s": c_pause_total,
                "child_pause_mean_s": c_pause_mean,
                "child_speech_ratio": c_speech_ratio,
            }

            rows.append(out_row)

        except Exception as e:
            print(f"\n Error on row {idx}: {e}")
            continue

    out_df = pd.DataFrame(rows)
    out_dir = os.path.dirname(OUTPUT_CSV)


    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n Saved prosodic features to: {OUTPUT_CSV}")
    print(f"   Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
