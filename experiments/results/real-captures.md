# Evaluation, captures photographs

Commit `1e99eb1`, format `0.3-pilot`, 2026-11-28. 6 photographs, 7 decoder variants. 1 photographs with `setup_ok = no` left out.

Means are over the runs that decoded. `MAE` compares the recovered gray values with the encoded ones; `spectral` compares the reconstructed audio with the encoded spectrogram, and the digital baseline is its lower bound (same Griffin-Lim settings, no printing).

## Stage-by-stage comparison

| Variant | Skipped | Options | Detected | Decoded | MAE gray | MAE dB | Spectral dB | Time s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| digital baseline | - | - | - | - | 0.00 | 0.00 | 0.71 | - |
| full | all | - | 6/6 | 6/6 | 9.54 | 2.25 | 2.18 | 5.8 |
| no-calibrate | calibrate | - | 6/6 | 6/6 | 57.11 | 13.44 | 16.25 | 3.0 |
| no-rectify | rectify+calibrate | - | 6/6 | 6/6 | 74.01 | 17.41 | 16.85 | 2.4 |
| no-detect | detect+rectify+calibrate | - | - | 6/6 | 74.01 | 17.41 | 16.85 | 1.1 |
| global-threshold | all | --threshold global | 6/6 | 6/6 | 9.54 | 2.24 | 2.18 | 4.8 |
| smooth-3 | all | --smooth 3 | 6/6 | 6/6 | 8.69 | 2.04 | 2.09 | 5.5 |
| samples-3 | all | --samples 3 | 6/6 | 6/6 | 9.23 | 2.17 | 2.17 | 5.7 |

## Detection per capture condition

| Condition | Photos | Detected, full | Detected, global-threshold |
| --- | --- | --- | --- |
| t00_l0 | 2 | 2/2 | 2/2 |
| t15p_l0 | 1 | 1/1 | 1/1 |
| t30p_l0 | 3 | 3/3 | 3/3 |

## Full pipeline, per photograph

| Photograph | Condition | Detected | MAE gray | MAE dB | Unmeasurable | Spectral dB | Baseline dB | Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20261126_c01_p1_t00_l0_01 | t00_l0 | yes | 10.04 | 2.36 | 0.0 % | 2.49 | 0.90 | - |
| 20261126_c01_p1_t15p_l0_01 | t15p_l0 | yes | 10.23 | 2.41 | 0.0 % | 2.60 | 0.90 | - |
| 20261126_c01_p1_t30p_l0_01 | t30p_l0 | yes | 8.78 | 2.07 | 0.0 % | 2.20 | 0.90 | - |
| 20261128_c02_p1_t00_l0_01 | t00_l0 | yes | 7.91 | 1.86 | 0.0 % | 1.28 | 0.52 | - |
| 20261128_c02_p1_t30p_l0_01 | t30p_l0 | yes | 8.79 | 2.07 | 0.0 % | 1.93 | 0.52 | - |
| 20261128_c02_p1_t30p_l0_02 | t30p_l0 | yes | 11.50 | 2.71 | 0.0 % | 2.60 | 0.52 | - |

## Time per stage, full pipeline

| Stage | Mean s |
| --- | --- |
| grayscale | 0.231 |
| detect | 1.355 |
| homography | 0.001 |
| rectify | 0.674 |
| calibrate | 2.731 |
| extract | 0.046 |
| magnitude | 0.006 |
| griffin-lim | 0.733 |
