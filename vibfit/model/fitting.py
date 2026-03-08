from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import hyperspy.api as hs
import numpy as np
from lmfit.models import PowerLawModel, PseudoVoigtModel

from .state import (
    BackgroundArea,
    BackgroundFitResult,
    EV_TO_CMINV,
    FitRegion,
    FitResultBundle,
    NOTEBOOK_PRESETS,
    PeakResult,
    SpectrumData,
    ev_to_cminv,
)


def load_signal(path: str):
    signal = hs.load(path)
    try:
        signal.set_signal_type("EELS")
    except Exception:
        pass
    return signal


def spectral_axis_from_signal(signal, spectrum_length: int) -> np.ndarray:
    signal_axes = list(getattr(signal.axes_manager, "signal_axes", []))
    if not signal_axes:
        return np.arange(spectrum_length, dtype=float)

    preferred_axes = []
    for axis in signal_axes:
        axis_values = np.asarray(axis.axis, dtype=float)
        if axis_values.size != spectrum_length:
            continue
        name = str(getattr(axis, "name", "")).lower()
        units = str(getattr(axis, "units", "")).lower()
        if any(token in name for token in ["energy", "loss", "eels", "ev", "cm"]):
            preferred_axes.append(axis_values)
            continue
        if any(token in units for token in ["ev", "mev", "cm", "1/cm"]):
            preferred_axes.append(axis_values)
            continue

    if preferred_axes:
        return preferred_axes[0]

    for axis in signal_axes:
        axis_values = np.asarray(axis.axis, dtype=float)
        if axis_values.size == spectrum_length:
            return axis_values

    return np.arange(spectrum_length, dtype=float)


def clone_region(region: FitRegion) -> FitRegion:
    return deepcopy(region)


def default_region() -> FitRegion:
    return clone_region(NOTEBOOK_PRESETS[0])


def _build_background_mask(x_cminv: np.ndarray, areas: list[BackgroundArea]) -> np.ndarray:
    mask = np.zeros_like(x_cminv, dtype=bool)
    for area in areas:
        x0 = min(float(area.x_min_cminv), float(area.x_max_cminv))
        x1 = max(float(area.x_min_cminv), float(area.x_max_cminv))
        mask |= (x_cminv >= x0) & (x_cminv <= x1)
    return mask


def _load_text_spectrum(path: Path) -> SpectrumData:
    for delimiter in (",", None):
        try:
            data = np.loadtxt(path, delimiter=delimiter)
            break
        except Exception:
            data = None
    if data is None:
        raise ValueError(f"Could not parse text spectrum file: {path}")
    if data.ndim == 1:
        x_ev = np.arange(data.size, dtype=float)
        intensity = np.asarray(data, dtype=float)
    elif data.shape[1] >= 2:
        x_ev = np.asarray(data[:, 0], dtype=float)
        intensity = np.asarray(data[:, 1], dtype=float)
    else:
        raise ValueError("Text spectrum must have either one column or x/y columns.")
    return SpectrumData(path=str(path), x_ev=x_ev, intensity=intensity, source_kind="text", title=path.name)


def _load_vxy_spectrum(path: Path) -> SpectrumData:
    try:
        data = np.loadtxt(path, skiprows=1)
    except Exception as exc:
        raise ValueError(f"Could not parse VXY spectrum file: {path}") from exc
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("VXY spectrum must contain energy, wavenumber, and intensity columns.")

    x_ev = np.asarray(data[:, 0], dtype=float)
    intensity = np.asarray(data[:, 2], dtype=float)
    return SpectrumData(path=str(path), x_ev=x_ev, intensity=intensity, source_kind="text", title=path.name)


def _load_numpy_spectrum(path: Path) -> SpectrumData:
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        if {"x_ev", "intensity"} <= set(data.files):
            x_ev = np.asarray(data["x_ev"], dtype=float)
            intensity = np.asarray(data["intensity"], dtype=float)
        elif {"energy", "intensity"} <= set(data.files):
            x_ev = np.asarray(data["energy"], dtype=float)
            intensity = np.asarray(data["intensity"], dtype=float)
        else:
            raise ValueError("NPZ spectrum requires x_ev/intensity or energy/intensity arrays.")
    else:
        array = np.load(path, allow_pickle=True)
        if array.ndim == 1:
            x_ev = np.arange(array.size, dtype=float)
            intensity = np.asarray(array, dtype=float)
        elif array.ndim == 2 and array.shape[0] == 2:
            x_ev = np.asarray(array[0], dtype=float)
            intensity = np.asarray(array[1], dtype=float)
        elif array.ndim == 2 and array.shape[1] >= 2:
            x_ev = np.asarray(array[:, 0], dtype=float)
            intensity = np.asarray(array[:, 1], dtype=float)
        else:
            raise ValueError("NPY spectrum requires 1D intensity or paired x/y arrays.")
    return SpectrumData(path=str(path), x_ev=x_ev, intensity=intensity, source_kind="numpy", title=path.name)


def _load_signal_spectrum(path: Path) -> SpectrumData:
    signal = load_signal(str(path))
    data = np.asarray(signal.data, dtype=float)
    if data.ndim == 1:
        intensity = data
    else:
        intensity = data.reshape((-1, data.shape[-1])).sum(axis=0)
    x_ev = np.asarray(spectral_axis_from_signal(signal, intensity.shape[0]), dtype=float)
    return SpectrumData(path=str(path), x_ev=x_ev, intensity=intensity, source_kind="signal", title=path.name)


def load_spectrum(path_str: str) -> SpectrumData:
    path = Path(path_str).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".vxy":
        return _load_vxy_spectrum(path)
    if suffix in {".csv", ".txt", ".dat"}:
        return _load_text_spectrum(path)
    if suffix in {".npy", ".npz"}:
        return _load_numpy_spectrum(path)
    return _load_signal_spectrum(path)


def fit_background(region: FitRegion, spectrum: SpectrumData) -> BackgroundFitResult:
    x_all = np.asarray(spectrum.x_cminv, dtype=float)
    y_all = np.asarray(spectrum.intensity, dtype=float)
    areas = list(region.background.fit_areas)
    if not areas:
        raise ValueError("Select one or more background fit areas first.")

    x_domain_min = min(min(area.x_min_cminv, area.x_max_cminv) for area in areas)
    x_domain_max = max(max(area.x_min_cminv, area.x_max_cminv) for area in areas)
    domain_mask = (x_all >= x_domain_min) & (x_all <= x_domain_max)
    if not np.any(domain_mask):
        raise ValueError("Background fit area span does not overlap the spectrum.")

    x_domain = x_all[domain_mask]
    y_domain = y_all[domain_mask]
    fit_mask = _build_background_mask(x_domain, areas)
    if fit_mask.sum() < 3:
        raise ValueError("Background fit areas do not contain enough points.")

    bg_model = PowerLawModel(prefix="bg_")
    params = bg_model.guess(y_domain[fit_mask], x=x_domain[fit_mask])
    result = bg_model.fit(y_domain[fit_mask], params, x=x_domain[fit_mask])
    background = np.asarray(result.eval(x=x_domain), dtype=float)
    y_bgsub = y_domain - background
    region_mask = (x_domain >= region.x_min_cminv) & (x_domain <= region.x_max_cminv)
    offset_source = y_bgsub[region_mask] if np.any(region_mask) else y_bgsub
    bgsub_offset = max(0.0, -float(np.nanmin(offset_source))) if offset_source.size else 0.0
    if bgsub_offset > 0.0:
        y_bgsub = y_bgsub + bgsub_offset
    return BackgroundFitResult(
        x_cminv=x_domain,
        y_raw=y_domain,
        background=background,
        y_bgsub=y_bgsub,
        bgsub_offset=float(bgsub_offset),
        area_mask=fit_mask,
        fit_report=result.fit_report(min_correl=0.5),
        success=bool(getattr(result, "success", True)),
        chisqr=float(result.chisqr),
        redchi=float(result.redchi),
        aic=float(result.aic),
        bic=float(result.bic),
    )


def build_fit(region: FitRegion, spectrum: SpectrumData, background_result: BackgroundFitResult) -> FitResultBundle:
    x_all = np.asarray(background_result.x_cminv, dtype=float)
    y_all = np.asarray(background_result.y_raw, dtype=float)
    bg_all = np.asarray(background_result.background, dtype=float)
    y_bgsub_all = np.asarray(background_result.y_bgsub, dtype=float)
    bgsub_offset = float(getattr(background_result, "bgsub_offset", 0.0))
    region_mask = (x_all >= region.x_min_cminv) & (x_all <= region.x_max_cminv)
    x = x_all[region_mask]
    y = y_all[region_mask]
    background = bg_all[region_mask]
    y_bgsub = y_bgsub_all[region_mask]
    if x.size < 5:
        raise ValueError("Selected fit region does not contain enough points.")
    if not region.peaks:
        raise ValueError("Add at least one peak before fitting.")

    model = None
    params = None
    for peak_spec in region.peaks:
        peak_model = PseudoVoigtModel(prefix=f"{peak_spec.name}_")
        peak_mask = (x >= peak_spec.guess_min_cminv) & (x <= peak_spec.guess_max_cminv)
        if peak_mask.sum() < 3:
            peak_mask = np.ones_like(x, dtype=bool)
        peak_params = peak_model.guess(y_bgsub[peak_mask], x=x[peak_mask])
        peak_params[f"{peak_spec.name}_amplitude"].set(
            value=peak_spec.amplitude.value,
            vary=peak_spec.amplitude.vary,
            min=peak_spec.amplitude.min,
            max=peak_spec.amplitude.max,
        )
        peak_params[f"{peak_spec.name}_center"].set(
            value=peak_spec.center.value,
            vary=peak_spec.center.vary,
            min=peak_spec.center.min,
            max=peak_spec.center.max,
        )
        peak_params[f"{peak_spec.name}_sigma"].set(
            value=max(peak_spec.sigma.value, 1e-9),
            vary=peak_spec.sigma.vary,
            min=peak_spec.sigma.min,
            max=peak_spec.sigma.max,
        )
        peak_params[f"{peak_spec.name}_fraction"].set(
            value=peak_spec.fraction.value,
            vary=peak_spec.fraction.vary,
            min=peak_spec.fraction.min,
            max=peak_spec.fraction.max,
        )
        if model is None:
            model = peak_model
            params = peak_params
        else:
            params += peak_params
            model += peak_model

    result = model.fit(y_bgsub, params, x=x)
    components = result.eval_components(x=x)
    peak_results: list[PeakResult] = []
    best_fit_bgsub = np.zeros_like(x, dtype=float)
    for peak_spec in region.peaks:
        component = np.asarray(components.get(f"{peak_spec.name}_", np.zeros_like(x)), dtype=float)
        best_fit_bgsub += component
        center_ev = float(result.best_values.get(f"{peak_spec.name}_center", peak_spec.center.value))
        sigma_ev = float(result.best_values.get(f"{peak_spec.name}_sigma", peak_spec.sigma.value))
        peak_results.append(
            PeakResult(
                name=peak_spec.name,
                center_ev=float(center_ev / EV_TO_CMINV),
                center_cminv=float(center_ev),
                amplitude=float(result.best_values.get(f"{peak_spec.name}_amplitude", peak_spec.amplitude.value)),
                sigma_ev=float(sigma_ev / EV_TO_CMINV),
                sigma_cminv=float(sigma_ev),
                fraction=float(result.best_values.get(f"{peak_spec.name}_fraction", peak_spec.fraction.value)),
                curve=component,
            )
        )

    return FitResultBundle(
        region_name=region.name,
        x_cminv=x,
        y_raw=y,
        best_fit=np.asarray(result.best_fit, dtype=float) + background - bgsub_offset,
        background=background,
        y_bgsub=y_bgsub,
        bgsub_offset=bgsub_offset,
        best_fit_bgsub=best_fit_bgsub,
        residual_raw=y - (np.asarray(result.best_fit, dtype=float) + background - bgsub_offset),
        residual_bgsub=y_bgsub - best_fit_bgsub,
        peaks=peak_results,
        fit_report=result.fit_report(min_correl=0.5),
        success=bool(getattr(result, "success", True)),
        chisqr=float(result.chisqr),
        redchi=float(result.redchi),
        aic=float(result.aic),
        bic=float(result.bic),
    )
