# Voxgram Project Plan

Digital Image Processing Team Project 02 · Team 29 SoundSight

This plan is the working version of the *Survey and Plan* report submitted on
3 November 2026. The report fixes the scope and the technical approach; this
document turns it into milestones and tracks the decisions still open. Items
marked **Proposed** are not in the report and need team agreement.

## 1. Goal

Record the magnitude information of a short voice signal as a printed grayscale
spectrogram card, and reconstruct intelligible speech from a smartphone
photograph of that card.

The central task is to recover useful spectrogram values after printing and
photography have changed image geometry and brightness. We implement the core
image-processing algorithms ourselves and measure their effect on extraction
accuracy and speech intelligibility.

## 2. Scope

**Core deliverable**

- Encoder: voice recording to printable card.
- Decoder: photograph of an intact card, taken with moderate tilt and
  illumination variation, to a WAV file.
- Stage-by-stage comparison showing what each correction step contributes.
- A report of the capture conditions under which the prototype succeeds or fails.

**Extensions, only after the core works**

- Partially damaged cards (known masks, simple interpolation)
- Heavy blur
- Real-time processing
- Graphical interface

**Out of scope**

- Smartphone application development
- Real-time camera streaming
- Correcting paper bending or strong lens distortion. The homography assumes a
  flat card.

## 3. Contribution relative to prior work

Printed spectrogram recovery is prior work (Aoki, 2023). Our contribution is:

1. a controlled card format with fiducial markers and calibration patches,
2. self-implemented marker detection, geometric correction and photometric
   correction,
3. evaluation on real smartphone photographs with stage-by-stage comparisons.

Griffin-Lim is used as a fixed reconstruction backend so that experiments
isolate changes in the image-processing front end.

## 4. System architecture

```
Encode   voice recording -> STFT magnitude -> grayscale mapping -> card layout -> print

Decode   smartphone photo -> marker detection -> geometric rectification
         -> brightness calibration -> spectrogram sampling -> magnitude recovery
         -> Griffin-Lim -> WAV playback
```

Processing and playback run in a Python program on a computer. The phone only
captures the image. The format parameters are in
[CARD_FORMAT.md](CARD_FORMAT.md).

### 4.1 Card design and detection

- Four high-contrast fiducial markers at known positions, an asymmetric
  orientation cue, white margins, and grayscale calibration patches around the
  spectrogram. The data region has no axes, labels or decoration.
- Grayscale conversion, then global and local-mean thresholding. An integral
  image accelerates the local mean. Global thresholding is the detection
  baseline.
- Two-pass connected-component labelling. Candidates are filtered by area,
  aspect ratio, fill ratio and relative arrangement.
- The orientation cue fixes marker order and prevents time or frequency
  reversal.
- Missing or ambiguous markers produce a clear detection failure, never an
  arbitrary audio output.

### 4.2 Geometric correction

- Four non-collinear marker correspondences define a planar homography.
- We build the normalized linear equations ourselves and use NumPy linear
  algebra only to solve them.
- Inverse mapping with bilinear interpolation resamples the grayscale
  photograph into canonical card coordinates.
- Coordinate normalization, denominator checks and rejection of degenerate
  quadrilaterals limit numerical failures.

### 4.3 Brightness and print correction

- Estimate spatially varying black and white references from distributed
  patches and normalize local brightness.
- Fit a monotonic piecewise-linear tone mapping from the intermediate gray
  patches.
- The objective is to recover the encoded grayscale values, not to increase
  visual contrast.
- Saturated glare and deep shadows cannot be inverted reliably and are flagged.

### 4.4 Extraction and limited restoration

- Sample the corrected data region back to the exact 257 x 376 grid, then apply
  the inverse intensity mapping.
- A small separable smoothing filter may be compared against no filtering to
  reduce print dots. Its settings are chosen on pilot cards. Smoothing can
  erase spectral detail, so it must earn its place in the ablation.
- Optional damage experiments use known masks and simple interpolation from
  the surrounding spectral structure.

## 5. Module ownership

| Area | Owner | Planned module (**Proposed** names) |
| --- | --- | --- |
| Audio format, encoder, digital baseline | 박창훈 | `format.py`, `encoder.py`, `audio_backend.py` |
| Evaluation scripts | 박창훈 | `evaluation.py` |
| Thresholding, connected components, marker detection | 강예원 | `detection.py` |
| Homography and resampling | Torval Ripel | `geometry.py` |
| Brightness calibration | Torval Ripel | `photometric.py` |
| ROI extraction, decoder integration | Cyprien Delrue | `extraction.py`, `decoder.py` |
| Experiment records | Cyprien Delrue | `experiments/` |

Every member also reviews another member's module. Review pairs are an open
decision (section 10).

### Interfaces between modules

The modules meet at three shared definitions. Changing any of them affects
more than one owner, so changes go through a pull request that all affected
owners review.

| Interface | Defined in | Consumers |
| --- | --- | --- |
| Audio, STFT and intensity parameters | `src/voxgram/format.py` | Encoder, decoder, evaluation |
| Canonical card coordinates: markers, patches, data region | [CARD_FORMAT.md](CARD_FORMAT.md), then `format.py` | Encoder layout, detection, geometry, photometric, extraction |
| Marker order returned by detection | To be defined with the detection module | Geometry |

## 6. Milestones

The report orders the work but sets no dates. Target dates are open
(section 10) and should be filled in from the course schedule.

| # | Milestone | Exit criterion | Target date |
| --- | --- | --- | --- |
| M0 | Repository bootstrap | Conventions, CI and format constants merged | Done with this commit |
| M1 | Digital baseline | Intelligible speech from an unprinted, unphotographed grayscale image | TBD |
| M2 | Card format frozen | Marker, cue and patch coordinates specified; first card printed | TBD |
| M3 | Front-facing photograph | Intelligible speech from a front-facing photograph of an undamaged card | TBD |
| M4 | Core deliverable | Works on intact cards with moderate tilt and illumination variation | TBD |
| M5 | Evaluation | Stage-by-stage ablation complete; success and failure conditions recorded | TBD |
| M6 | Final report and presentation | Submitted | TBD |
| X | Extensions | Damage, heavy blur, real-time, GUI, as time allows | After M5 |

**Gate after M1 and M3.** If either fails, adjust duration, grid density,
dynamic range or physical print size before working on difficult capture
cases. M1 does not depend on any image-processing module, so it starts first
and in parallel with the detection and geometry work.

## 7. Evaluation plan

From the report:

- **Digital baseline.** Reconstruct from the unmodified encoded grayscale
  image. This separates Griffin-Lim artifacts from photographic degradation.
- **Stage-by-stage comparison.** Compare images and reconstructed speech
  before and after each correction stage.
- **Ablation.** Keep the unchanged original grayscale photograph so a failed
  correction can be diagnosed and compared with an ablated pipeline.
- **Baselines inside the pipeline.** Global thresholding against local-mean
  thresholding; smoothing against no smoothing.
- **Capture conditions.** Record the printer, paper, print resolution and
  phone settings for every capture, and report where the prototype succeeds
  and fails.
- **Processing time** is logged per stage.

**Proposed** metrics, to be confirmed by the team:

| Quantity | Candidate metric |
| --- | --- |
| Extraction accuracy | Mean absolute error between recovered grayscale `p` and encoded `q` on the 257 x 376 grid; the same error expressed in dB |
| Detection | Marker detection success rate per capture condition |
| Geometry | Reprojection error of marker centers in canonical coordinates |
| Speech intelligibility | Listening comparison against the digital baseline; an objective measure if the team agrees on one |

## 8. Library boundary

Implementation is in Python on Windows, in a Visual Studio-compatible project.

| Component | Implementation or permitted support |
| --- | --- |
| Image detection | Own grayscale conversion, thresholds, integral image, components and marker logic |
| Geometry | Own equation assembly, coordinate handling, inverse warp and bilinear interpolation |
| Brightness and ROI | Own patch statistics, tone mapping, spatial interpolation and grid sampling |
| Optional restoration | Own small convolution filters and masked interpolation |
| Numerical support | NumPy arrays, arithmetic and linear-system or SVD solver; no ready-made vision pipeline |
| Image and audio I/O | OpenCV image read/write; audio file read/write and playback utilities |
| Audio backend | Librosa STFT and Griffin-Lim only, as disclosed support; fixed settings across conditions |
| Excluded shortcuts | No OpenCV `threshold`, `findContours`, `connectedComponents`, `findHomography`, `warpPerspective`, `resize`, `equalizeHist` or `inpaint` |

CI runs `tools/check_library_boundary.py`, which rejects any OpenCV call
outside image read/write.

## 9. Risks

| Risk | Mitigation |
| --- | --- |
| Speech is unintelligible even from a clean digital image | M1 gate. Adjust duration, grid density or dynamic range before any printing work. |
| Print resolution cannot carry the grid | M3 gate. Adjust physical print size or grid density. At 600 dpi one grid cell is about 7 printer dots wide. |
| Griffin-Lim artifacts are mistaken for image-processing errors | Always compare against the digital baseline with identical backend settings. |
| Halftone dots corrupt the recovered values | Test a small separable smoothing filter against no filtering on pilot cards. |
| Markers missed or misordered | Fail clearly; the orientation cue fixes the order; global thresholding as a baseline to compare against. |
| Glare or deep shadow | Flag saturated regions instead of inverting them. |
| Paper bending or lens distortion | Out of scope. Keep the card flat and away from the image edges. |
| Slow pure-Python loops | Vectorize with NumPy arrays; log processing time. |
| Encoder and decoder disagree on the format | Single source of truth in `format.py`; format changes need review by all affected owners. |

## 10. Open decisions

| # | Decision | Needed by |
| --- | --- | --- |
| 1 | Target dates for M1 to M6, from the course schedule | Now |
| 2 | Review pairs: who reviews whose module | Before the first module PR |
| 3 | Marker shape and size, orientation cue, patch count and gray levels, exact canonical coordinates | M2 |
| 4 | Final values of duration, grid density, dynamic range and print size | After the M1 and M3 gates |
| 5 | Evaluation metrics and the intelligibility procedure | Before M5 |
| 6 | Where raw photographs and voice recordings are stored. They are kept out of this public repository by default. | Before the first capture session |
| 7 | Printer, paper and phone used for the reference captures | M2 |
| 8 | Repository license | Before the final submission |

## 11. Expected outcomes

- A prototype that converts a short voice recording into a printable
  spectrogram card and reconstructs speech from a smartphone photograph.
- A set of self-implemented image-processing modules for card detection,
  geometric correction, brightness calibration and spectrogram extraction.
- Evidence of how each processing stage affects extraction accuracy and speech
  intelligibility, and of which capture conditions suit printed spectrograms.
- A basis for applications such as printed voice-message cards and educational
  material connecting visual patterns with sound.

## 12. References

1. D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform," *IEEE Trans. Acoustics, Speech, and Signal Processing*, vol. 32, no. 2, pp. 236-243, 1984. DOI: 10.1109/TASSP.1984.1164317
2. A. Kawamura, "On sound signal processing in image to sound mapping technique," *Applied Acoustics*, vol. 117, Part A, pp. 1-11, 2017. DOI: 10.1016/j.apacoust.2016.10.014
3. N. Aoki, "Speech Restoration of Spectrogram Images Printed in a Document 'Visible Speech' Published in 1947," *IEICE Technical Report*, vol. 123, no. 95, EA2023-6, pp. 12-15, July 2023. https://ken.ieice.org/ken/paper/20230702yCVh/eng/
4. N. Aoki, "A survey of an inverse halftoning technique toward speech restoration of printed spectrogram images," *Acoustical Science and Technology*, vol. 44, no. 4, pp. 336-339, 2023. DOI: 10.1250/ast.44.336
5. Librosa, "librosa.griffinlim," version 0.11.0 documentation. https://librosa.org/doc/0.11.0/generated/librosa.griffinlim.html
