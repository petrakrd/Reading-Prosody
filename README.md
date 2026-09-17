## Pipeline order

1. **`01_extract_prosodic_features.py`**
   Reads `imitation_baseline_alignment_table.csv`, loads each child's
   reading (PRT) and imitation (PIT) recordings, and extracts pitch
   (F0), intensity, and pause features, including reading-vs-imitation
   contour similarity (DTW). Outputs `prosodic_features_imitation_RAW.csv`.

2. **`02_compute_derived_features.py`**
   Takes the Phase 1 output and computes the final derived metrics
   used in the manuscript (duration ratio / Tempo Stability, pause
   intrusions / Pausal Intrusions, and supporting F0 measures).
   Outputs `prosodic_features_imitation_ENHANCED.csv`.

3. See **`codebook.md`** for the full mapping between column names in
   the code and the terminology used in the manuscript (e.g.,
   `duration_ratio` = Tempo Stability, `F0_contour_dtw` = Melody Error,
   `int_contour_dtw` = Rhythm Error).

## Design

Each child's own reading-task recording is compared against that same
child's own imitation-task recording (their individual acoustic
baseline). All "imitation_*"
columns refer to that child's PIT recording; all "child_*" columns
refer to their PRT (reading) recording.

## Files in this repository

| File | Description |
|---|---|
| `01_extract_prosodic_features.py` | Phase 1: acoustic feature extraction |
| `02_compute_derived_features.py` | Phase 2: derived metrics (Tempo Stability, Pausal Intrusions) |
| `codebook.md` | Column name ↔ manuscript terminology mapping |
| `requirements.txt` | Python dependencies |
| `imitation_baseline_alignment_table.csv` | Reading/imitation file pairing per child and sentence (illustrative paths; audio not included) |
| `final_analysis_dataset_anonymized.csv` | De-identified, participant-level dataset with only the variables used in the reported analyses (age, gender, PA, RAN, Reading_Fluency, Tempo_Stability, Pausal_Intrusions, Melody_Error, Rhythm_Error) |
| `prosodic_reading_task_stimuli.csv` | The 29 PRT sentences, with word/syllable counts, intonation type, and sentence type |
| `prosodic_imitation_task_stimuli.csv` | The 29 PIT sentences (same structure; presentation order differs slightly, per Method) |

## Data availability

Raw audio recordings (.wav) — both children's task recordings and the
adult reference stimuli recorded by the two speech-language
therapists — are not included in this repository. `imitation_baseline_alignment_table.csv`
and the two stimuli files are included to document task
structure, but their audio paths are illustrative and will not
resolve without the original audio corpus. Speaker
identities in the stimuli files have been anonymized to `SLP_A` /
`SLP_B`.

The final, anonymized, participant-level feature and outcome dataset
used in the reported analyses (age, gender, PA, RAN, Reading Fluency,
Tempo Stability, Pausal Intrusions, Melody Error, Rhythm Error) is
provided as `final_analysis_dataset_anonymized.csv`.

## Requirements

See `requirements.txt`. Tested with Python 3.13.
