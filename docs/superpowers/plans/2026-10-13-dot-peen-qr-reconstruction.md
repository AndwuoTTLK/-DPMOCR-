# Dot-peen QR reconstruction implementation plan

**Goal:** Make the desktop app decode the supplied isolated steel dot-peen QR image after the existing fast decoders fail.

**Architecture:** Keep the current whole-image and text-guided WeChat QR paths. Add a final, bounded fallback for nearly square red-dot-on-green images: estimate the QR module lattice from the three finder patterns, sample dot color at module centers, render a clean binary QR, and accept content only when OpenCV validates the reconstructed code.

**Tech stack:** Python 3.12, OpenCV, NumPy, SciPy differential evolution. No network call at runtime.

**Spec:** User request in this conversation and `research_artifacts/steel_dot_qr_source_2026-10-13.png` with independently confirmed content `8=H2543=234=01L-C2=/LG=12AH36==S4`.

## Tasks

- [x] Add a failing end-to-end test for the supplied image and a no-code square-image control.
- [x] Implement bounded lattice fitting and reconstruction in a separate `dot_peen_qr.py` module, returning no result unless a QR decoder confirms it.
- [x] Call the fallback from `ImageRecognizer.recognize_qr` only after existing decoders fail; add the SciPy dependency.
- [x] Run the new tests, all existing tests, the full 30-image corpus, and GUI startup; record runtime and scope limitations.

## Review focus

- This method is for isolated, approximately square red-dot codes on green steel; other image styles should skip it quickly.
- Scanning failures must return no QR result, preserving OCR fallback.
- Successful output must come from QR error-correcting decode, not pattern-based guessing.
- The older five decodable samples must keep their original payloads.
