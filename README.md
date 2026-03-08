# vibfit

`vibfit` is a Qt desktop application for constrained vibEELS peak fitting with `lmfit`.


## Install

```bash
pip install vibfit
```

## Run

```bash
vibfit
```

## Development build

```bash
python -m build
```

## Notes

- fitting is performed in `cm^-1`
- vibrational background fitting uses a `PowerLaw` model
- peak names are editable and are used as the `lmfit` prefixes
