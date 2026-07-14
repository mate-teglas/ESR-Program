# ESR Analysis Suite — filtered, individual-measurement version

## What changed

- Every `.dat` file is fitted independently. Angle clusters are used only for
  navigation and inclusion/exclusion; they never define an averaged fit.
- UP and DOWN remain separate fitting series.
- Temperature and frequency are parsed from each header and can be filtered on
  the Data tab.
- The current ESR format is supported directly:
  - `TB(Sample)=9.99@10.00` stores measured temperature `9.99 K` and groups the
    file under the nominal temperature `10.00 K`.
  - `APSyn=17.9785 GHz` together with `Anapico x6` gives an effective frequency
    of approximately `108 GHz`.
  - Folder names such as `108 GHz` or `10K` are fallback metadata sources.
- Filtered files can be copied or moved into folders such as:

  ```text
  T_10K/F_108GHz/UP/
  T_10K/F_108GHz/DOWN/
  ```

- Fitting warm starts and domain tracking are isolated by
  `(temperature, frequency, direction)`.
- A tracking reference may contain 1, 2, 3, or 4 fitted domains. For a
  two-domain analysis, fit and name the two lines, then use that spectrum as
  the reference.
- The material-name field on the Data tab is used in analysis and browser
  figure titles.
- A user-selected fit-output directory is used for fit parameters, fitted
  curves, sessions, and saved analysis figures.
- Fit CSV files can be loaded to recreate parameter plots. Use `.esrsession`
  when the interactive browser, residuals, and full fitted curves are needed.

## Recommended workflow

1. Select the raw-data folder, set prefix/extension, and press **Load Data**.
2. Choose one temperature and one frequency, then press **Apply filters**.
3. Optionally use **Create separated folders** in `copy` mode. Use `move` only
   when you intentionally want to reorganize the original data.
4. Choose the fit-output directory.
5. On Fitting, choose UP or DOWN and set Min/Max components. Use `2/2` to force
   a two-domain fit or `2/4` for automatic weak-domain discovery.

   The **Maximum Lorentzians** value is a hard upper bound. The **Two dominant
   only** preset sets Minimum = 2 and Maximum = 2, so residual discovery cannot
   add a third or fourth line. The GUI also includes **Automatic 2–4** and
   **Force four** presets.
6. Press **Fit All Angles**. Every matching file is fitted separately, even if
   multiple files have identical angle, direction, temperature, and frequency.
7. Review/correct fits in the interactive browser. A reference with the same
   number of domains as the intended analysis is accepted.

   Domain renaming has four propagation scopes:

   - **current** changes only the displayed measurement;
   - **after →** changes the displayed measurement and every subsequent
     measurement reached with the right arrow;
   - **← before** changes the displayed measurement and every preceding
     measurement reached with the left arrow;
   - **everything** changes the complete matching series.

   Propagation is restricted to the same temperature, frequency, and sweep
   direction. It swaps the two tracked domain identities without changing any
   fitted line parameters, and missing domains remain gaps. With **Lock
   assignment** enabled, propagated labels are protected from later automatic
   re-tracking.
8. Save an `.esrsession` to preserve arrays, review states, manual corrections,
   locks, and tracking. Exported fit CSVs are suitable for parameter replots.

## Fit output

With automatic saving enabled, Fit All creates:

```text
FIT_out/
  fit_results.csv
  fitted_measurements/
    0001_<source>_<direction>_<angle>deg_fit.csv
    ...
```

Each individual curve CSV contains magnetic field, measured signal, total fit,
and residual, plus source filename, angle, direction, temperature, and
frequency metadata.

## Run

```bash
cd "/path/to/ESR_Analysis_Suite_filtered_individual"
python3 main.py
```

Install dependencies once if required:

```bash
python3 -m pip install numpy scipy matplotlib pandas
```
