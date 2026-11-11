<!-- Title format: <type>(<scope>): <subject>   e.g. feat(detect): add marker filter -->

## Summary

<!-- What changed and why. -->

Closes #

## How it was tested

<!-- Unit tests, synthetic inputs, real captures. Attach before/after images or numbers where useful. -->

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `python tools/check_library_boundary.py` passes
- [ ] `pytest` passes
- [ ] Format values come from `voxgram.format`, not literals
- [ ] No raw photographs, recordings or generated outputs are committed
- [ ] If the format changed: `FORMAT_VERSION` bumped and `docs/CARD_FORMAT.md` updated
