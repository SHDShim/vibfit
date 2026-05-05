# vibfit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20018218.svg)](https://doi.org/10.5281/zenodo.20018218)

`vibfit` is a Qt desktop application for constrained vibEELS peak fitting using `lmfit`. It is particularly useful for vibrational spectra with broad peaks.

Current release: `0.5.0`

## Install

Install the current release from PyPI:

```bash
pip install vibfit
```

For development from a local checkout:

```bash
python -m pip install -e .
```

## Run

```bash
vibfit
```

## How to use

1. Launch the app with `vibfit`.
2. In the `File` tab, click `Load spectrum (vxy)` and open a supported spectrum file. `vibfit` currently supports `*.vxy`. Support for additional formats is planned; contact `shdshim` if you need another file type.
3. Use the toolbar to zoom to the spectral range you want to analyze.
4. In the `Background` tab, click `Select area` and mark one or more background-only regions on the top plot. Click `Fit background` when you are done. To add another region, click `Select area` again.
5. In the `PeakFit` tab, click `Set fit range`, then drag across the bottom plot to define the fitting window. Use `Clear fit range` to restore the full spectrum.
6. In the `PeakFit` tab, click `Pick peaks` and add peaks within the active fit range in the bottom panel. To add a peak, hold `Shift`, click near the left side of the peak FWHM, drag to the right side, and release. Repeat for each peak. To remove a peak, right-click near it.
7. Edit peak names and parameter bounds in the table as needed, then click `Fit region`.
8. Review the fit results table and report. Use `Save to section` to store the current fitted region in the session and clear the active fit before continuing to the next region.
9. Use `Save XLS` in the `Sections` tab to export all saved sections as `.json` and `.xlsx` files, with one Excel sheet per section. Use `Export NPY, PDF, and PNG` to export plot arrays together with a reproduction Python script and rendered figures.
10. Click `Save` in the toolbar to create a session backup. Backups are stored next to the source spectrum in a `*-vibfit` directory and can be restored from the `File` tab.

Useful controls:

- `Zoom in` focuses the view on the active fit region.
- `Zoom out` returns the view to the full spectrum.
- `Yspec` rescales the plots based on the current spectrum view.
- `Yadj` automatically adjusts the visible y-range.
- The `Plot` tab lets you enter exact top and bottom y-axis limits.
- To review a previously saved section, open the `Sections` tab, select a section, and click `Set current`. The saved fit results will be plotted again.

## Development build

```bash
python -m build
```

## Citation

Citation metadata is provided in `CITATION.cff`.

- Latest-release DOI badge: `10.5281/zenodo.20018217`
- Version `0.5.0` DOI: `10.5281/zenodo.20018218`

## License

`vibfit` is distributed under the BSD 3-Clause License. See `LICENSE`.

## Notes

- fitting is performed in `cm^-1`
- vibrational background fitting uses a `PowerLaw` model
- peak names are editable and are used as `lmfit` prefixes
