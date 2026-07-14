# ESR Program

Desktop analysis program for amplitude-modulated ESR measurements. This
repository contains the last stable **non-temperature-comparison** version of
the application.

## Main features

- Fits every `.dat` measurement independently; repeated measurements are never
  averaged.
- Parses angle, sweep direction, sample temperature, and effective microwave
  frequency from file metadata.
- Filters measurements by temperature, frequency, direction, angle, and file.
- Fits one to four non-derivative mixed absorptive/dispersive Lorentzian
  resonances plus a cubic baseline.
- Provides hard minimum/maximum Lorentzian controls, including a two-dominant-
  domains mode.
- Tracks domain identities separately for each temperature, frequency, and
  sweep direction.
- Supports manual center selection, constrained refitting, domain renaming,
  locking, review states, and reference spectra with one to four domains.
- Exports fit parameters, individual fitted curves, review images, and complete
  `.esrsession` files.
- Can copy or move filtered raw measurements into metadata-based folders.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

## Recommended workflow

1. Load a folder containing ESR `.dat` measurements.
2. Select one temperature and frequency and apply the desired filters.
3. Choose UP or DOWN fitting direction.
4. Set the minimum and maximum number of Lorentzians.
5. Run **Fit All Angles**. Every selected file receives its own fit.
6. Review and correct the fits in the interactive browser.
7. Set a trusted reference spectrum and verify domain tracking.
8. Save an `.esrsession` and export the fitted parameters and curves.

For the detailed filtering, tracking, and output behavior, see
[README_FILTERS_AND_INDIVIDUAL_FITS.md](README_FILTERS_AND_INDIVIDUAL_FITS.md).

## Scientific model

Each physical resonance is represented by a non-derivative mixture of an
absorptive and a dispersive Lorentzian. The total model is the sum of the fitted
resonances and a cubic polynomial baseline. Integrated absorptive intensity is
calculated as

```text
I = |pi * gamma * A * cos(theta)|
```

where `gamma` is the Lorentzian HWHM and `A*cos(theta)` is the absorptive
coefficient.

## Repository scope

Raw experimental data, fit sessions, virtual environments, and generated fit
outputs are intentionally excluded from Git. This repository contains the
program source and documentation only.
