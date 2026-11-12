# Voxgram

**Voice reconstruction from printed spectrogram images.**
Digital Image Processing Team Project 02 · Team 29 SoundSight

Voxgram stores the magnitude of a short voice recording as a printed grayscale
spectrogram card, then reconstructs the speech from a smartphone photograph of
that card. Decoding uses only the card and a fixed public format; the original
audio file is never consulted.

The hard part is not the audio. Printing and photography change the geometry
and brightness of the image, and the project is about recovering usable
spectrogram values in spite of that. The image-processing pipeline is
implemented by hand; existing libraries are used only for I/O, linear algebra
and the audio backend.

> **Status:** the encoder and the digital baseline (recording to grayscale
> image and back, without printing) are implemented. Card layout, marker
> detection, geometric and brightness correction are not.
> Code freeze is Thursday 26 November 2026; source code submission is
> Saturday 28 November 2026.
> See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for milestones.

## Pipeline

```
Encode   voice recording -> STFT magnitude -> grayscale mapping -> card layout -> print

Decode   smartphone photo -> marker detection -> geometric rectification
         -> brightness calibration -> spectrogram sampling -> magnitude recovery
         -> Griffin-Lim -> WAV playback
```

A binary working image is used only for marker detection. The grayscale image
keeps its intensity values through rectification and extraction.

## Pilot format

| Item | Value |
| --- | --- |
| Audio | Mono, 16 kHz, padded to 3.0 s (48,000 samples) |
| STFT | 512-point FFT, Hann window, hop 128, `center=True`, constant padding |
| Magnitude grid | 257 frequency bins x 376 frames |
| Axes | Linear frequency, low frequency at the bottom, time left to right |
| Dynamic range | Peak-normalized, -60 to 0 dB, darker means larger magnitude |
| Card | About 150 x 100 mm, with a 110 x 75 mm spectrogram region |

These values are provisional until the first pilot validates them. The full
specification is in [docs/CARD_FORMAT.md](docs/CARD_FORMAT.md), and the shared
constants live in [src/voxgram/format.py](src/voxgram/format.py).

## Getting started

Requires Python 3.10 or newer. Development is done on Windows.

```powershell
git clone https://github.com/pch4947/Voxgram.git
cd Voxgram
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Encode a recording, or run the full digital round trip:

```powershell
python -m voxgram.encoder recording.wav -o spectrogram.png
python -m voxgram.baseline recording.wav -o outputs/baseline/recording
```

The recording must be 3.0 s or shorter; add `--truncate` to keep only the
first 3.0 s of a longer one. The baseline writes the spectrogram image and
three WAV files to compare by ear: the original, Griffin-Lim from the exact
magnitude, and Griffin-Lim from the 8-bit image.

Run the checks that CI runs:

```powershell
ruff check .
ruff format --check .
python tools/check_library_boundary.py
pytest
```

## Library boundary

The course allows supporting library functions that do not perform the main
idea. We draw the line as follows.

| Component | Rule |
| --- | --- |
| Image detection | Own grayscale conversion, thresholds, integral image, connected components, marker logic |
| Geometry | Own equation assembly, coordinate handling, inverse warp, bilinear interpolation |
| Brightness and ROI | Own patch statistics, tone mapping, spatial interpolation, grid sampling |
| Optional restoration | Own small convolution filters and masked interpolation |
| Numerical support | NumPy arrays, arithmetic, linear-system or SVD solver |
| Image and audio I/O | OpenCV image read/write only; audio file read/write and playback utilities |
| Audio backend | Librosa STFT and Griffin-Lim, with fixed settings across all conditions |

**Not allowed:** OpenCV `threshold`, `findContours`, `connectedComponents`,
`findHomography`, `warpPerspective`, `resize`, `equalizeHist`, `inpaint`, or
any other ready-made vision routine.

`tools/check_library_boundary.py` enforces the OpenCV part of this rule in CI.

## Repository layout

```
.github/            Issue and PR templates, CI workflow
docs/               Project plan, card format, git convention
src/voxgram/        Python package
  format.py           Shared format constants
  intensity.py        Magnitude <-> grayscale mapping, both directions
  audio_backend.py    Audio I/O, Librosa STFT and Griffin-Lim
  imageio.py          Image read/write (the only OpenCV use)
  encoder.py          Recording -> spectrogram image
  baseline.py         Digital round trip without printing
tests/              Unit tests
tools/              Repository tooling (library boundary check)
```

The remaining pipeline modules will be added under `src/voxgram/` by their
owners as the milestones progress. The planned module split is in
[docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md#5-module-ownership).

## Team

| Member | Primary responsibility |
| --- | --- |
| 박창훈 | Audio format, encoder and digital baseline; evaluation scripts |
| 강예원 | Thresholding, connected components and marker detection |
| Torval Ripel | Homography, resampling and brightness calibration |
| Cyprien Delrue (team leader) | ROI extraction, decoder integration and experiment records |

Every member implements code, reviews another member's module, and takes part
in the experiments and the final presentation.

## Documentation

- [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md): scope, milestones, evaluation, risks
- [docs/CARD_FORMAT.md](docs/CARD_FORMAT.md): audio, STFT, intensity and card layout specification
- [docs/GIT_CONVENTION.md](docs/GIT_CONVENTION.md): branches, commits, pull requests
- [CONTRIBUTING.md](CONTRIBUTING.md): day-to-day workflow and code rules

## References

1. D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform," *IEEE Trans. Acoustics, Speech, and Signal Processing*, vol. 32, no. 2, pp. 236-243, 1984. DOI: 10.1109/TASSP.1984.1164317
2. A. Kawamura, "On sound signal processing in image to sound mapping technique," *Applied Acoustics*, vol. 117, Part A, pp. 1-11, 2017. DOI: 10.1016/j.apacoust.2016.10.014
3. N. Aoki, "Speech Restoration of Spectrogram Images Printed in a Document 'Visible Speech' Published in 1947," *IEICE Technical Report*, vol. 123, no. 95, EA2023-6, pp. 12-15, July 2023.
4. N. Aoki, "A survey of an inverse halftoning technique toward speech restoration of printed spectrogram images," *Acoustical Science and Technology*, vol. 44, no. 4, pp. 336-339, 2023. DOI: 10.1250/ast.44.336
5. Librosa, "librosa.griffinlim," version 0.11.0 documentation. https://librosa.org/doc/0.11.0/generated/librosa.griffinlim.html
