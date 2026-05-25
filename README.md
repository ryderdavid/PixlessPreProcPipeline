# Pixless PreProc Pipeline

Astrophotography preprocessing pipeline that does not use PixInsight.

“Pixless” here means doing calibration, stacking, and related prep outside of PixInsight—open-source and scriptable tooling instead.

## XisfPrep

This repo vendors [XisfPrep](https://github.com/JonathanMaccollum/XisfPrep) as a git submodule at `XisfPrep/`—an F# CLI for batch preprocessing XISF astro images (calibrate, align, integrate, and more).

```bash
git clone --recurse-submodules https://github.com/ryderdavid/PixlessPreProcPipeline.git
# or, after a plain clone:
git submodule update --init --recursive
```
