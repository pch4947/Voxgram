"""Shared format constants for the Voxgram encoder and decoder.

This module is the single source of truth for the values specified in
docs/CARD_FORMAT.md. Import from here instead of repeating the numbers.

Changing anything in this file changes the card format. Follow the format
change rule in docs/GIT_CONVENTION.md.
"""

FORMAT_VERSION = "0.3-pilot"

# Audio
SAMPLE_RATE = 16_000  # Hz, mono
DURATION_S = 3.0
NUM_SAMPLES = round(SAMPLE_RATE * DURATION_S)  # 48,000

# STFT. The same settings are used for analysis and for Griffin-Lim.
N_FFT = 512
HOP_LENGTH = 128
WINDOW = "hann"
CENTER = True
PAD_MODE = "constant"

# Magnitude grid. Rows are frequency bins, columns are frames.
NUM_FREQ_BINS = N_FFT // 2 + 1  # 257
NUM_FRAMES = 1 + NUM_SAMPLES // HOP_LENGTH  # 376, valid because CENTER is True
GRID_SHAPE = (NUM_FREQ_BINS, NUM_FRAMES)

# Intensity encoding. Magnitude is peak-normalized, converted to amplitude dB
# and floored at DB_FLOOR. 0 dB maps to GRAY_BLACK and DB_FLOOR to GRAY_WHITE,
# so darker means larger magnitude.
DB_FLOOR = -60.0
GRAY_BLACK = 0
GRAY_WHITE = 255

# Canonical card coordinates, in millimetres. The origin is the top-left
# corner of the card, x runs right and y runs down. Draft layout, see
# "Canonical coordinates" in docs/CARD_FORMAT.md.
CARD_WIDTH_MM = 150.0
CARD_HEIGHT_MM = 100.0
TARGET_PRINT_DPI = 600

# Spectrogram data region: top-left corner and size. Image row 0 is at the top.
DATA_X_MM = 20.0
DATA_Y_MM = 12.5
DATA_WIDTH_MM = 110.0
DATA_HEIGHT_MM = 75.0

# Fiducial markers: a black square frame, a white gap and a black core, all
# concentric squares given by their side lengths. The orientation marker has a
# smaller core, so the frame-to-core area ratio tells the two kinds apart.
MARKER_SIZE_MM = 7.0
MARKER_GAP_MM = 5.0
MARKER_CORE_MM = 3.0
MARKER_CUE_CORE_MM = 1.5

# Marker centres in the order used for homography correspondences:
# top-left, top-right, bottom-right, bottom-left (clockwise on the card).
MARKER_CENTERS_MM = ((8.0, 8.0), (142.0, 8.0), (142.0, 92.0), (8.0, 92.0))
CUE_MARKER_INDEX = 2  # the bottom-right marker is the orientation marker

# Calibration patches: the same 11 gray levels on each side of the data
# region. Level 0 is solid black; level 255 is unprinted paper.
PATCH_LEVELS = (0, 26, 51, 77, 102, 128, 153, 179, 204, 230, 255)

_STEPS = range(len(PATCH_LEVELS))
# Top and bottom patches are 8 x 6 mm at a 10 mm pitch; side patches are
# 6 x 5 mm at a 6.5 mm pitch. The ramps run clockwise from dark to light, so
# every level appears on all four sides and black sits near every corner.
_TOP = tuple((21.0 + 10.0 * k, 3.5, 29.0 + 10.0 * k, 9.5) for k in _STEPS)
_RIGHT = tuple((135.0, 15.0 + 6.5 * k, 141.0, 20.0 + 6.5 * k) for k in _STEPS)
_BOTTOM = tuple((121.0 - 10.0 * k, 90.5, 129.0 - 10.0 * k, 96.5) for k in _STEPS)
_LEFT = tuple((9.0, 80.0 - 6.5 * k, 15.0, 85.0 - 6.5 * k) for k in _STEPS)

# Every patch as (x0, y0, x1, y1, level), with (x0, y0) the top-left corner
# and (x1, y1) the bottom-right corner.
CALIBRATION_PATCHES = tuple(
    (*rect, level)
    for side in (_TOP, _RIGHT, _BOTTOM, _LEFT)
    for rect, level in zip(side, PATCH_LEVELS, strict=True)
)
