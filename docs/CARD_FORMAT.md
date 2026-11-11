# Voxgram Card Format

**Format version:** `0.1-pilot`. Provisional until the first pilot validates it.

This is the shared specification that the encoder and the decoder must agree
on. The numeric values here are mirrored in
[`src/voxgram/format.py`](../src/voxgram/format.py); code imports them from
there and never repeats them as literals. To change the format, follow the
rule in [GIT_CONVENTION.md](GIT_CONVENTION.md#format-changes).

## 1. Audio

| Parameter | Value |
| --- | --- |
| Channels | Mono |
| Sample rate | 16,000 Hz |
| Duration | 3.0 s, which is 48,000 samples |
| Short recordings | Padded to 3.0 s |
| Silent recordings | Rejected by the encoder |

Handling of recordings longer than 3.0 s is not yet specified.

## 2. STFT

| Parameter | Value |
| --- | --- |
| FFT size | 512 |
| Window | Hann |
| Hop length | 128 |
| Centering | `center=True` |
| Padding mode | Constant |

Resulting magnitude grid:

- Frequency bins: 512 / 2 + 1 = **257**, linear from 0 to 8,000 Hz
- Frames: 1 + floor(48,000 / 128) = **376**

Reconstruction must use exactly the same STFT settings.

## 3. Image orientation

| Axis | Convention |
| --- | --- |
| Horizontal | Time, left to right. Column 0 is the first frame. |
| Vertical | Frequency, low frequency at the bottom. The top image row is bin 256. |
| Pixel value | 8-bit grayscale. 0 is black, 255 is white. |

## 4. Intensity encoding

Let `A` be the STFT magnitude divided by its maximum over the whole recording,
so `A` lies in [0, 1].

```
D = max(20 * log10(A), -60)          amplitude in dB, in [-60, 0]
q = round(-255 * D / 60)             stored grayscale, in [0, 255]
```

Darker means larger magnitude: the peak (`D = 0`) prints black (`q = 0`), and
everything at or below -60 dB prints white (`q = 255`).

## 5. Intensity decoding

Let `p` be the recovered grayscale value in [0, 255] after geometric and
photometric correction.

```
D_hat = -60 * p / 255
A_hat = 10 ** (D_hat / 20)
```

`A_hat` is a normalized linear magnitude in [0.001, 1].

What is lost:

- **Phase.** Estimated by Griffin-Lim.
- **Absolute level.** The magnitude is peak-normalized before encoding.
- **Values below -60 dB.** Clipped to the floor.
- **Precision.** One gray level is 60 / 255, about 0.235 dB.

## 6. Physical layout

| Item | Value |
| --- | --- |
| Card size | About 150 x 100 mm |
| Spectrogram region | 110 x 75 mm |
| Paper | Matte white |
| Print resolution | Target 600 dpi if available |

Derived cell size: 110 mm / 376 frames and 75 mm / 257 bins both come to about
0.29 mm per cell, which is about 7 printer dots at 600 dpi.

Record the actual printer, paper, print settings and phone settings with every
capture.

## 7. Card elements

The card carries:

- four high-contrast fiducial markers at known positions,
- one asymmetric orientation cue,
- white margins,
- grayscale calibration patches around the spectrogram: black and white
  references distributed across the card, plus intermediate gray levels,
- the spectrogram data region, with no axes, labels or decorative graphics.

### Canonical coordinates: not yet specified

The following must be fixed before the first card is printed (milestone M2).
Until then the encoder layout, detection, geometry, photometric and extraction
modules cannot be integrated.

| Item | Status |
| --- | --- |
| Coordinate origin, axis directions and unit | TBD |
| Marker shape, size and the four center positions | TBD |
| Marker order convention (for the homography correspondences) | TBD |
| Orientation cue shape and position | TBD |
| Calibration patch count, size, positions and gray levels | TBD |
| Data region corner positions | TBD |
| Margin widths | TBD |

## 8. Version history

| Version | Date | Change |
| --- | --- | --- |
| `0.1-pilot` | 2026-11-11 | Initial pilot values from the Survey and Plan report |
