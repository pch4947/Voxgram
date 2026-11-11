# Git Convention

All commit messages, branch names, issues, pull requests and code comments are
written in English.

## Branches

We use GitHub Flow: short-lived branches off `main`, merged by pull request.

- `main` always passes CI. Nobody pushes to it directly.
- One branch per issue. Delete the branch after merging.
- Rebase on `main` or merge `main` in before asking for review if the branch
  has fallen behind.

Branch names:

```
<type>/<issue-number>-<short-description>
```

```
feat/12-marker-detection
fix/27-homography-degenerate-quad
docs/5-card-coordinates
exp/31-tilt-sweep
```

Use lowercase and hyphens. Leave out the issue number only when there is no
issue.

## Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Subject**

- Imperative mood: "add", not "added" or "adds".
- Lowercase first letter, no trailing period.
- 72 characters or fewer.

**Body**, optional

- Explain what changed and why, not how. The diff shows how.
- Wrap at 72 characters.

**Footer**, optional

- `Closes #12` to close an issue, `Refs #12` to link one.
- `BREAKING CHANGE: <description>` for format changes (see below).

### Types

| Type | Use for |
| --- | --- |
| `feat` | A new capability in the pipeline or tools |
| `fix` | A bug fix |
| `docs` | Documentation only |
| `test` | Adding or correcting tests |
| `refactor` | Code change that neither fixes a bug nor adds a capability |
| `perf` | A change that makes the code faster |
| `exp` | Experiment scripts, capture records and result tables |
| `style` | Formatting only, no change in behavior |
| `build` | Dependencies and packaging |
| `ci` | CI configuration |
| `chore` | Anything else that does not touch source or tests |

### Scopes

The scope names the part of the project that changed. It is optional but
encouraged.

| Scope | Area |
| --- | --- |
| `format` | Shared format constants and card specification |
| `encoder` | Audio to card |
| `detect` | Thresholding, connected components, marker detection |
| `geometry` | Homography, inverse warp, interpolation |
| `photo` | Brightness and tone calibration |
| `extract` | ROI sampling and magnitude recovery |
| `decoder` | End-to-end decoder integration |
| `audio` | Librosa backend wrapper, audio I/O |
| `eval` | Evaluation scripts and metrics |

### Examples

```
feat(detect): add two-pass connected-component labelling

Uses union-find over 8-connected foreground pixels. Components are
returned with area and bounding box so the marker filter can use them.

Closes #12
```

```
fix(geometry): reject quadrilaterals with collinear markers
```

```
exp(eval): record tilt sweep on pilot card 02
```

```
docs(format): specify marker positions in canonical coordinates
```

## Pull requests

- Title uses the same format as a commit subject. It becomes the commit
  message on `main`.
- Link the issue with `Closes #<number>` in the description.
- Keep pull requests small: one module change or one experiment.
- At least one approval from a team member other than the author. Changes to a
  module should be reviewed by someone who does not own it.
- CI must pass.
- Merge with **Squash and merge**, so `main` has one commit per pull request.
- The author merges after approval and deletes the branch.

Open a draft pull request early if you want feedback before the work is done.

## Format changes

The encoder and the decoder only work together if they agree on the format.
A change to `src/voxgram/format.py` or `docs/CARD_FORMAT.md` therefore has
extra rules:

1. Change both files in the same pull request.
2. Bump `FORMAT_VERSION` and add a row to the version history in
   `docs/CARD_FORMAT.md`.
3. Add `BREAKING CHANGE:` to the commit footer if cards printed under the old
   version can no longer be decoded.
4. Request review from every owner whose module reads the changed values.

## Issues

- One issue per unit of work. Use the templates.
- Assign yourself when you start.
- Labels: `feat`, `bug`, `docs`, `experiment`, plus one area label matching the
  scopes above.

## What not to commit

- Raw photographs and voice recordings. This repository is public, and a voice
  recording is personal data. Store them where the team agrees and reference
  them from the experiment record. `data/` is ignored by Git.
- Generated outputs: reconstructed WAV files, rectified images, debug dumps.
  `outputs/` is ignored by Git.
- Virtual environments, caches and IDE settings.

Small fixtures that tests need go in `tests/data/` and should stay under about
1 MB each.

## Tags

Tag the commit used for each submitted deliverable:

```
git tag -a v0.1-pilot -m "First pilot card"
git push origin v0.1-pilot
```
