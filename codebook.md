# Codebook

Maps the column names produced by the prosodic feature-extraction
pipeline to the terminology used in the paper

All "imitation_*" columns refer to a given child's own recording from
the Prosodic Imitation Task (PIT). All "child_*" columns refer to that
same child's recording from the Prosodic Reading Task (PRT). Every
comparison is within-child (reading vs. that child's own
imitation baseline)

## Final analysis variables 

| Column (code) | Manuscript term | Notes |
|---|---|---|
| `duration_ratio` (Phase 2 output) | **Tempo Stability** | `dur_child_s / dur_imitation_s`. Values > 1.0 indicate temporal slowing during reading relative to the imitation baseline. |
| `pause_intrusions` (Phase 2 output) | **Pausal Intrusions** | `child_pause_count − imitation_pause_count`, using a 150 ms minimum duration and a 4% RMS energy threshold (`PAUSE_MIN_DUR`, `PAUSE_RMS_FRAC` in Phase 1). |
| `F0_contour_dtw` (Phase 1 output) | **Melody Error** | Path-normalized DTW cost between z-scored, 50-point-resampled F0 contours (reading vs. imitation). |
| `int_contour_dtw` (Phase 1 output) | **Rhythm Error** | Path-normalized DTW cost between z-scored, 50-point-resampled intensity (amplitude envelope) contours (reading vs. imitation). |

## Other variables (from the 3DM-H battery; not part of this code pipeline)

| Column | Manuscript term |
|---|---|
| `PA` | Phonological Awareness (Phoneme Deletion subtest, age-standardized score) |
| `RAN` | Rapid Automatized Naming (age-standardized speed score; higher = faster) |
| `Reading_Fluency` | Reading Fluency (age-standardized score, 3DM-H) |
| `age` | Child age in years |
| `gender` | Child sex (boy/girl) |

## Exploratory variables (from Phase 1 output, not used in paper as outputs)

| Column | Description |
|---|---|
| `dur_child_s`, `dur_imitation_s` | Total duration (s) of the reading / imitation recording |
| `dur_child_speech_s`, `dur_imitation_speech_s` | Duration excluding detected pauses |
| `child_F0_mean/sd/min/max/range`, `imitation_F0_mean/sd/min/max/range` | Descriptive F0 (pitch) statistics per recording |
| `child_F0_voiced_ratio`, `imitation_F0_voiced_ratio` | Proportion of voiced frames |
| `child_int_mean_dB`, `imitation_int_mean_dB` | Mean intensity (dB) |
| `F0_contour_corr`, `int_contour_corr` | Pearson correlation between reading and imitation contours (exploratory; not used in reported analyses) |
| `child_pause_count/total_s/mean_s`, `imitation_pause_count/total_s/mean_s` | Pause counts and durations per recording |
| `child_speech_ratio`, `imitation_speech_ratio` | Proportion of the recording that is speech (non-pause) |

## Phase 2 (derived features) columns  

| Column | Description |
|---|---|
| `child_F0_z`, `imitation_F0_z`, `F0_z_diff` | Z-scored mean F0 and their difference (exploratory) |
| `child_final_pitch_drop`, `imitation_final_pitch_drop`, `final_drop_diff` | F0 range (max − min) per recording and their difference (exploratory) |
| `pause_total_diff`, `pause_mean_diff` | Additional pause-based difference scores (exploratory; `pause_intrusions` is the metric reported as Pausal Intrusions) |

## Stimulus materials (`prosodic_reading_task_stimuli.csv`, `prosodic_imitation_task_stimuli.csv`)

| Column | Description |
|---|---|
| `Order` | Presentation order within the task |
| `Item_id` | Sentence identifier (`pros_1`–`pros_30`), consistent across both files |
| `Sentence` | The Hungarian sentence text |
| `word_count`, `syllable_count` | Sentence length |
| `intonation_type` | Hungarian label for the target intonation pattern: `lebego` = **Sustained**, `ereszkedo` = **Descending**, `eso` = **Falling** (manuscript terminology) |
| `sentence_type` | Syntactic structure: declarative / question / imperative / exclamation |
| `audiofile` | Relative path to the adult reference stimulus recording (speaker anonymized as `SLP_A`/`SLP_B`; audio itself not included) |
| `adult_file_name` | Generic stimulus ID used elsewhere in the pipeline |
