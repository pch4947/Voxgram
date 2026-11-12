"""Shared format constants for the Voxgram encoder and decoder.

This module is the single source of truth for the values specified in
docs/CARD_FORMAT.md. Import from here instead of repeating the numbers.

Changing anything in this file changes the card format. Follow the format
change rule in docs/GIT_CONVENTION.md.
"""

FORMAT_VERSION = "0.2-pilot"

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

# Physical layout, in millimetres. The card size is approximate.
CARD_WIDTH_MM = 150.0
CARD_HEIGHT_MM = 100.0
DATA_WIDTH_MM = 110.0
DATA_HEIGHT_MM = 75.0
TARGET_PRINT_DPI = 600

# Marker, orientation cue, calibration patch and data region coordinates are
# not specified yet. See "Canonical coordinates" in docs/CARD_FORMAT.md.
