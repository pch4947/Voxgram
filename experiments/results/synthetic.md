# Evaluation, synthetic photographs

Commit `e404ddd`, format `0.3-pilot`, 2026-11-21. 21 photographs, 7 decoder variants.

Means are over the runs that decoded. `MAE` compares the recovered gray values with the encoded ones; `spectral` compares the reconstructed audio with the encoded spectrogram, and the digital baseline is its lower bound (same Griffin-Lim settings, no printing).

## Stage-by-stage comparison

| Variant | Skipped | Options | Detected | Decoded | MAE gray | MAE dB | Spectral dB | Time s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| digital baseline | - | - | - | - | 0.00 | 0.00 | 0.78 | - |
| full | all | - | 21/21 | 21/21 | 1.18 | 0.28 | 0.80 | 3.3 |
| no-calibrate | calibrate | - | 21/21 | 21/21 | 34.37 | 8.09 | 7.37 | 1.6 |
| no-rectify | rectify+calibrate | - | 21/21 | 21/21 | 67.83 | 15.96 | 14.59 | 1.0 |
| no-detect | detect+rectify+calibrate | - | - | 21/21 | 67.83 | 15.96 | 14.59 | 0.7 |
| global-threshold | all | --threshold global | 21/21 | 21/21 | 1.18 | 0.28 | 0.80 | 3.1 |
| smooth-3 | all | --smooth 3 | 21/21 | 21/21 | 5.25 | 1.24 | 1.19 | 3.3 |
| samples-3 | all | --samples 3 | 21/21 | 21/21 | 1.23 | 0.29 | 0.78 | 3.4 |

## Detection per capture condition

| Condition | Photos | Detected, full | Detected, global-threshold |
| --- | --- | --- | --- |
| front | 3 | 3/3 | 3/3 |
| roll-15 | 3 | 3/3 | 3/3 |
| tilt-mild | 3 | 3/3 | 3/3 |
| tilt-moderate | 3 | 3/3 | 3/3 |
| dim | 3 | 3/3 | 3/3 |
| gradient | 3 | 3/3 | 3/3 |
| combined | 3 | 3/3 | 3/3 |

## Full pipeline, per photograph

| Photograph | Condition | Detected | MAE gray | MAE dB | Marker error mm | Cell error mm | Unmeasurable | Spectral dB | Baseline dB | Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| libri1_front | front | yes | 0.44 | 0.10 | 0.000 | 0.000 | 0.0 % | 0.75 | 0.73 | - |
| libri1_roll-15 | roll-15 | yes | 0.46 | 0.11 | 0.001 | 0.001 | 0.0 % | 0.72 | 0.73 | - |
| libri1_tilt-mild | tilt-mild | yes | 0.46 | 0.11 | 0.040 | 0.038 | 0.0 % | 0.72 | 0.73 | - |
| libri1_tilt-moderate | tilt-moderate | yes | 0.48 | 0.11 | 0.037 | 0.036 | 0.0 % | 0.76 | 0.73 | - |
| libri1_dim | dim | yes | 0.78 | 0.18 | 0.001 | 0.001 | 0.0 % | 0.76 | 0.73 | - |
| libri1_gradient | gradient | yes | 2.61 | 0.61 | 0.001 | 0.001 | 0.0 % | 0.74 | 0.73 | - |
| libri1_combined | combined | yes | 2.97 | 0.70 | 0.030 | 0.030 | 0.0 % | 0.79 | 0.73 | - |
| libri2_front | front | yes | 0.45 | 0.11 | 0.001 | 0.001 | 0.0 % | 0.63 | 0.62 | - |
| libri2_roll-15 | roll-15 | yes | 0.38 | 0.09 | 0.001 | 0.001 | 0.0 % | 0.59 | 0.62 | - |
| libri2_tilt-mild | tilt-mild | yes | 0.46 | 0.11 | 0.040 | 0.038 | 0.0 % | 0.63 | 0.62 | - |
| libri2_tilt-moderate | tilt-moderate | yes | 0.44 | 0.10 | 0.037 | 0.036 | 0.0 % | 0.60 | 0.62 | - |
| libri2_dim | dim | yes | 0.73 | 0.17 | 0.001 | 0.001 | 0.0 % | 0.68 | 0.62 | - |
| libri2_gradient | gradient | yes | 2.84 | 0.67 | 0.001 | 0.001 | 0.0 % | 0.70 | 0.62 | - |
| libri2_combined | combined | yes | 2.65 | 0.62 | 0.028 | 0.028 | 0.0 % | 0.62 | 0.62 | - |
| libri3_front | front | yes | 0.52 | 0.12 | 0.001 | 0.001 | 0.0 % | 0.98 | 0.98 | - |
| libri3_roll-15 | roll-15 | yes | 0.54 | 0.13 | 0.001 | 0.001 | 0.0 % | 0.96 | 0.98 | - |
| libri3_tilt-mild | tilt-mild | yes | 0.55 | 0.13 | 0.040 | 0.038 | 0.0 % | 1.04 | 0.98 | - |
| libri3_tilt-moderate | tilt-moderate | yes | 0.55 | 0.13 | 0.037 | 0.036 | 0.0 % | 0.99 | 0.98 | - |
| libri3_dim | dim | yes | 0.86 | 0.20 | 0.001 | 0.001 | 0.0 % | 0.98 | 0.98 | - |
| libri3_gradient | gradient | yes | 2.84 | 0.67 | 0.001 | 0.001 | 0.0 % | 0.98 | 0.98 | - |
| libri3_combined | combined | yes | 2.84 | 0.67 | 0.029 | 0.029 | 0.0 % | 1.13 | 0.98 | - |

## Time per stage, full pipeline

| Stage | Mean s |
| --- | --- |
| grayscale | 0.008 |
| detect | 0.308 |
| homography | 0.001 |
| rectify | 0.506 |
| calibrate | 1.852 |
| extract | 0.036 |
| magnitude | 0.004 |
| griffin-lim | 0.582 |
