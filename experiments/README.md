# Experiments

How we photograph printed cards and record the results, so that captures from
different days, phones and printers can be compared.

Items marked **Proposed** are not yet agreed by the team. Change them in a pull
request once the team has decided; the protocol is only useful if everybody
follows the same one.

## Contents

| File | Purpose |
| --- | --- |
| `README.md` | This protocol: capture procedure, naming scheme, storage rule |
| `TEMPLATE.md` | Session record. Copy it to `records/<session>.md` for every session |
| `captures.csv` | One row per photograph and processing run. The table all comparisons are made from |
| `records/` | Filled-in session records (created with the first record) |

Raw photographs and voice recordings are **not** in this repository. See
[section 5](#5-where-raw-data-is-stored).

## 1. Rules that apply to every capture

1. Record everything that can differ between two captures: card, printer,
   paper, print settings, phone, camera settings, lighting, distance, tilt and
   the commit of the code. The record is what makes two captures comparable.
2. Change **one** condition at a time. A tilt sweep keeps lighting and distance
   fixed; a lighting sweep keeps the phone front-facing.
3. Never delete a photograph because the result is bad. A failed capture is a
   result. Mark it in `captures.csv` instead.
4. Never edit, crop, rotate or filter an original photograph. Processing starts
   from the file exactly as the phone saved it.
5. Never write on the front of a card. Identify cards by a mark on the back.

## 2. Printing the card

Record these in the session record (`TEMPLATE.md`) for every printed card.

| Setting | Rule |
| --- | --- |
| Scale | **Actual size, 100 %.** Turn off "fit to page" and any automatic margins. A wrong scale changes the physical cell size |
| Resolution | The highest the printer offers; target 600 dpi or more (`TARGET_PRINT_DPI` in `format.py`) |
| Colour mode | Grayscale or black ink only, if the printer has the option. Mixed colour inks give a colour cast |
| Paper | Matte white. Glossy paper gives glare. Note brand and weight (g/m²) |
| Image smoothing | Turn off any "enhance", "smooth", "photo optimisation" or "toner save" feature |
| After printing | Let the card dry or cool for at least 10 minutes, then measure it |

Measure the printed card with a ruler: the card size (about 150 x 100 mm) and
the width and height of the data region (about 110 x 75 mm). Write the measured
values in the record. Anything more than about 1 mm from the specification
means the scale was wrong; print again.

Handle cards by the edges, keep them flat (no folding, no rolling) and store
them in a folder. Paper bending is out of scope for the project, so a curled
card is a bad capture.

## 3. Capture protocol

### 3.1 Set-up

- **Surface:** a plain, matte, dark or mid-gray surface with no pattern. The
  card has white margins, so a dark surface gives the markers a clean border.
- **Card flat:** the card lies flat on the surface. If it curls, hold it with
  small pieces of tape on the outer white margin, away from the markers,
  calibration patches and data region. Do not press it under glass: glass
  causes glare.
- **Phone support:** use a phone stand or books for the front-facing shots if
  you can. If you hold the phone, rest your elbows on the table.
- **Shadows:** check that neither your hand nor the phone casts a shadow on the
  card, unless a shadow is the condition being tested (L3 below). At short
  distances the phone itself is the usual cause.

### 3.2 Camera settings

Use the phone's own camera app in photo mode, and keep the same settings for
every capture of the same phone.

| Setting | Value |
| --- | --- |
| Lens | Main (1x) camera. No zoom. Turn off automatic switching to a macro or ultra-wide lens |
| Resolution | The highest resolution saved as JPEG. Record the megapixels |
| File format | **JPEG**, not HEIC. On iPhone: Settings, Camera, Formats, Most Compatible |
| Flash | Off |
| HDR, night mode, portrait mode, filters, beauty, scene optimiser, AI enhancement | Off |
| Live Photo / motion photo | Off |
| Location tagging | Off. Photographs carry GPS data otherwise |
| Focus and exposure | Tap the centre of the card to focus, then lock focus and exposure (long press on most phones) |
| Exposure compensation | 0. Lower it only if the white margin of the card looks blown out in the preview, and record the value |
| Shutter | Use the 3 second timer or a volume button, to avoid shake |

The white paper must not be clipped to pure white in the photograph. A clipped
white cannot be calibrated, so this is the setting that matters most.

### 3.3 Framing and distance

Distance is fixed by framing, not by a number, because phones differ:

- The whole card, including its white margins, is in the frame.
- The card is oriented with its long side along the long side of the image
  (landscape).
- The card fills about 70 to 85 % of the width of the image.
- At least 5 % of the image width remains clear on every side of the card, also
  when the card is tilted. Lens distortion is worst at the image edges.
- **Check (Proposed):** the data region must be at least 1,500 px wide in the
  photograph, which is 4 px per cell across 376 cells. Any current phone at its
  full resolution does this with the framing above.

Record the approximate distance from the lens to the centre of the card, in
centimetres (ruler or a marked string). It is a record, not a setting.

### 3.4 Conditions

The core deliverable (milestone M4) covers a front-facing card, moderate tilt
and illumination variation. The codes below are used in file names and in
`captures.csv`.

**Tilt.** The card lies flat on the table and the phone is tilted above it.
Read the angle from the phone's level or inclinometer app (iPhone: Measure,
Level; Android: any bubble-level app). When the phone lies parallel to the
table, the level reads 0 degrees.

| Code | Name | Meaning |
| --- | --- | --- |
| `t00` | Front-facing | Camera looks straight down onto the card. Level reads 0 degrees, within 3 degrees. The card edges are parallel to the image edges, within 5 degrees |
| `t15` | Mild tilt | Phone tilted 15 degrees from front-facing (Proposed) |
| `t30` | Moderate tilt | Phone tilted 30 degrees from front-facing (Proposed). This is what "moderate tilt" means in the project plan |
| `t45` | Strong tilt | 45 degrees. Stress test, outside the core deliverable (Proposed) |

For `t15`, `t30` and `t45` the tilt axis is part of the code, as a letter:

- `p` for pitch: the top or bottom edge of the card is nearer to the camera.
- `y` for yaw: the left or right edge of the card is nearer to the camera.

Example: `t30p` is a 30 degree pitch tilt. Tilt about one axis at a time.
Combined tilts are allowed as extra experiments, described in `notes`.

**Illumination.** The reference condition is the one every other condition is
compared against.

| Code | Name | How |
| --- | --- | --- |
| `l0` | Reference | Even, diffuse light. No visible shadow, gradient or glare. For example daylight from a window without direct sun, plus ceiling light |
| `l1` | Dim | The same arrangement with about half the light, for example some lights off |
| `l2` | Gradient | One lamp from the side at a low angle, so one side of the card is clearly brighter than the other |
| `l3` | Partial shadow | A soft shadow of an object (a book, a hand) across part of the card. Stress test |
| `l4` | Glare | Light placed so that a reflection from the paper appears in the photograph. Stress test |

Measure the light with a lux app on the phone, with the sensor facing the light
source and held at the card. For `l2` and `l3`, note the darkest and brightest
reading on the card. Lux apps are rough; they are good enough to rank
conditions, not to report absolute values.

### 3.5 What to take (Proposed)

Take **3 photographs** for every condition, numbered `01` to `03`. Frame and
focus again for each one, so that the three show the repeatability of your
hand and of the phone's autofocus.

Suggested minimum per card and phone:

| Series | Conditions | Photos |
| --- | --- | --- |
| Reference | `t00` with `l0` | 3 |
| Tilt | `t15p`, `t15y`, `t30p`, `t30y`, all with `l0` | 12 |
| Illumination | `t00` with `l1`, `l2`, `l3` | 9 |
| Optional stress | `t45p`, `t45y`, `l4` | 9 |

Shoot the reference series first. If the reference series does not decode, the
other series cannot be interpreted yet.

### 3.6 Checklist

Before the first photograph:

- [ ] Session record copied from `TEMPLATE.md` and the set-up fields filled in
- [ ] Phone settings as in 3.2, location tagging off
- [ ] Card measured and flat, mark on the back
- [ ] Lighting checked: no unwanted shadow or glare

After every photograph:

- [ ] Card entirely in frame, 5 % clear on every side
- [ ] White margin not blown out, image not blurred
- [ ] If anything was wrong, keep the photograph and set `setup_ok` to `no`

## 4. Naming scheme

All names are lowercase ASCII with no spaces: some of us work in folders with
Korean names, and tools on Windows do not handle every path equally well.

### Photographs

```
<date>_<card>_<phone>_<tilt>_<light>_<nn>.jpg
```

| Field | Format | Example |
| --- | --- | --- |
| `date` | `YYYYMMDD` | `20261117` |
| `card` | `c` and two digits, written on the back of the card | `c01` |
| `phone` | `p` and a digit, registered in the session record | `p1` |
| `tilt` | code from 3.4 | `t00`, `t30p` |
| `light` | code from 3.4 | `l0` |
| `nn` | repeat number, two digits | `01` |

Example: `20261117_c01_p1_t30p_l0_02.jpg`

Keep the file as the phone saved it. If the phone saves another extension,
convert nothing and keep it, but change the phone settings (3.2).

The file name without extension is the **capture ID**. It is the key in
`captures.csv`.

If an experiment varies another factor, such as distance, add it as an extra
field before `nn` (`d25` for 25 cm) and add the field to this table in the same
pull request.

### Voice recordings

```
<date>_<speaker>_<nn>.wav
```

`speaker` is a code (`sp01`, `sp02`, ...), never a name. The list that links
codes to people stays in the private storage, not in this repository. Example:
`20261116_sp01_01.wav`.

### Cards

A card is named `c01`, `c02`, and so on, in the order they are printed. The
session record states for each card the format version, the source image that
was printed and the commit of the code that rendered it. The source image is
stored privately (section 5), because it contains the voice.

### Processing outputs

Decoder outputs go in `outputs/<capture_id>/<commit>/`, with the 7-character
commit hash of the code that produced them. `outputs/` is ignored by Git.

```
python -m voxgram.decoder data/photos/<date>/<capture_id>.jpg \
    -o outputs/<capture_id>/<commit>/voice.wav \
    --save-intermediates outputs/<capture_id>/<commit>/steps
```

## 5. Where raw data is stored

This resolves open decision 6 in
[`docs/PROJECT_PLAN.md`](../docs/PROJECT_PLAN.md#10-open-decisions). It was
taken by the owner of issue #12. Anyone who disagrees says so in the pull
request that adds this file, and the section is changed there.

Why raw data stays out of this repository: it is public, and a voice recording
is personal data. A card and a photograph of it also contain the voice, because
decoding them gives the speech back. Cards, photographs and recordings are
therefore all treated as personal data.

**Decision:**

- A **private shared Google Drive folder** named `Voxgram-data`, owned by the
  team leader and shared by invitation with the four team members' accounts.
  Never set it to "anyone with the link".
- The link to the folder is shared in the team's private chat. **It is not
  written in this repository**, because anybody could open it.
- Each member keeps a local copy of what they need in `data/` at the root of
  their clone. `data/` is ignored by Git, so it cannot be committed by mistake.

Why two places:

- The **cloud folder** is the single shared source. It is where a photograph
  goes after a session, so that every member sees the same files, and it is the
  backup if a laptop is lost. Git cannot play this role here, because the
  repository is public.
- The **local copy** is what the code reads. The decoder and the evaluation
  scripts open files by path on the member's own disk, and they do that faster
  and more reliably from a normal folder than from a cloud one.
- The cloud folder is the reference. If the two differ, the cloud folder is
  right. Nobody edits files in `data/`.
- To get files, download the folder or the sub-folder you need from Drive and
  unzip it into `data/`, keeping the structure below. Do **not** use Google
  Drive for desktop to sync a folder inside your clone: it can lock files and
  slows Git down.

Structure, the same in the cloud folder and in `data/`:

```
Voxgram-data/
  recordings/      <date>_<speaker>_<nn>.wav
  cards/           c01_source.png, c01_source.wav, ...
  photos/<date>/   <capture ID>.jpg
  KEY.txt          speaker code to person, private
```

Rules:

- Upload the photographs of a session the same day, and add the rows to
  `captures.csv` before leaving the session. A photograph without a row, or a
  row without a photograph, cannot be used.
- Voices are recorded only with the speaker's consent, and only team members'
  voices are used, or speech without personal content (counting, a read text).
- A speaker can ask for their recordings, cards and photographs to be deleted at
  any time.
- Printed cards made from a real voice are stored with the team, not left in
  shared or public places.
- Photographs have location tagging switched off (3.2). Check the file
  properties before uploading if in doubt.
- Raw data is kept until the final report is graded; the team then deletes the
  recordings and the cards and photographs made from them, unless the speaker
  agrees to keep them.

Small synthetic fixtures that tests need, such as the card from the card
renderer, are not personal data and may go in `tests/data/`, under about 1 MB
each, as the git convention says.

## 6. Records

### Session record

One Markdown file per capture session, from `TEMPLATE.md`, saved as
`experiments/records/<date>_session-<nn>.md`. It holds what stays the same
during the session: cards and how they were printed, phone and camera settings,
lighting set-up, commit and format version, deviations, and where the files are.

### `captures.csv`

One row per photograph and per processing run. The columns are:

| Column | Meaning |
| --- | --- |
| `capture_id` | File name without extension (section 4) |
| `date` | `YYYY-MM-DD` |
| `session` | Session record name, without extension, for example `20261117_session-01` |
| `card` | Card code, for example `c01` |
| `format_version` | Value of `FORMAT_VERSION` the card was printed with |
| `phone` | Phone code, for example `p1`. The model is in the session record |
| `tilt_deg` | Measured tilt angle in degrees, `0` for front-facing |
| `tilt_axis` | `none`, `pitch` or `yaw` |
| `roll_deg` | Approximate in-plane rotation of the card in the image, `0` if aligned. Estimate by eye |
| `light` | Lighting code (`l0` to `l4`) |
| `lux_min`, `lux_max` | Lowest and highest lux reading on the card. Empty if not measured |
| `distance_cm` | Approximate distance from the lens to the card |
| `ev_comp` | Exposure compensation, `0` by default |
| `setup_ok` | `yes` if the conditions matched this protocol, `no` if not. A `no` row is kept but left out of success rates |
| `commit` | 7-character hash of the code that produced the result. Empty until processed |
| `stages` | `all`, or the stages switched off with the decoder's `--skip`, separated by `+`: `calibrate`, `rectify+calibrate` or `detect+rectify+calibrate`. Skipping `rectify` also skips `calibrate`, and skipping `detect` skips all three. Other options that were changed (`--threshold global`, `--smooth 3`, `--card-dpi`) go in `notes` |
| `detect_ok` | `yes`, `no` or `ambiguous`: did marker detection succeed |
| `mae_gray` | Mean absolute error between the recovered gray values `p` and the encoded values `q` over the 257 x 376 grid, in gray levels. Empty if unknown |
| `mae_db` | The same error in dB (1 gray level is about 0.235 dB) |
| `intelligibility` | Listening score, 0 to 5 (below). Empty if not listened to |
| `notes` | Free text. No commas, or put the text in double quotes |

Intelligibility scale (**Proposed**, until the team decides the procedure,
open decision 5): 0 noise only; 1 speech-like sound, no words; 2 a few words;
3 most words, with effort; 4 all words, noisy; 5 as good as the digital
baseline.

Rules for the file:

- The first line is the header and is never changed. Add columns only in a pull
  request that also updates this table.
- Fill the conditions at the session. Fill `commit` and the result columns
  after processing.
- To process the same photograph again with other code or other stages, copy
  its row, and change `commit`, `stages` and the results. A result is
  identified by `capture_id`, `commit` and `stages` together. Never overwrite
  an earlier result.
- Empty means "not known yet", not zero.
- One row per line, with a newline at the end of the file. Two people adding
  rows in parallel can then resolve a merge conflict by keeping both sets of
  lines.

Example row (made up, to show the format):

```
20261117_c01_p1_t30p_l0_02,2026-11-17,20261117_session-01,c01,0.3-pilot,p1,30,pitch,0,l0,,,28,0,yes,a1b2c3d,all,yes,6.4,1.5,3,"card slightly curled at the top right"
```

## 7. Working with Git

- Commit records and scripts with the type `exp`, for example
  `exp: record the reference series on card c01`. Scope is optional.
- One pull request per session or per experiment, with the session record,
  the new `captures.csv` rows, and `Closes #<issue>` where an experiment issue
  exists.
- The pull request template asks you to confirm that no raw photographs,
  recordings or generated outputs are committed. Check `git status` before
  `git commit`: `.jpg`, `.wav` and `data/` must not appear.
- Open an issue from the *Experiment* template for each planned session, and
  link the session record and the commit in its **Result** section.
