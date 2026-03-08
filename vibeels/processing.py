from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import hyperspy.api as hs
import numpy as np
from lmfit.models import LinearModel, PseudoVoigtModel


EV_TO_CMINV = 8065.54429


@dataclass
class ZeroLossFit:
    center_ev: float
    amplitude: float
    sigma: float
    fraction: float
    background_slope: float
    background_intercept: float
    fit_x: np.ndarray
    fit_y: np.ndarray
    best_fit: np.ndarray
    calibrated_axis: np.ndarray


@dataclass
class MapProcessingResult:
    intensity_image: np.ndarray
    display_image: np.ndarray
    masked_image: np.ndarray
    selection_mask: np.ndarray
    selected_pixel_count: int
    selected_spectra: np.ndarray
    selected_zlp_centers_ev: np.ndarray
    summed_spectrum: np.ndarray
    energy_axis_raw: np.ndarray
    energy_axis_calibrated: np.ndarray
    zero_loss_fit: ZeroLossFit


@dataclass
class StackProcessingResult:
    detector_image_raw: np.ndarray
    aligned_stack: np.ndarray
    summed_spectrum: np.ndarray
    energy_axis_raw: np.ndarray
    energy_axis_calibrated: np.ndarray
    zero_loss_fit: ZeroLossFit


def load_signal(path: str):
    signal = hs.load(path)
    try:
        signal.set_signal_type("EELS")
    except Exception:
        pass
    return signal


def axis_from_signal(signal, axis_index: int = -1) -> np.ndarray:
    axes = signal.axes_manager.signal_axes
    if axes:
        axis = axes[axis_index]
        return np.asarray(axis.axis, dtype=float)

    size = signal.data.shape[axis_index]
    return np.arange(size, dtype=float)


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


def ensure_range(start: int, stop: int, limit: int) -> Tuple[int, int]:
    start = max(0, min(int(start), limit - 1))
    stop = max(start + 1, min(int(stop), limit))
    return start, stop


def rectangle_mask(
    height: int,
    width: int,
    x_start: int,
    x_stop: int,
    y_start: int,
    y_stop: int,
) -> np.ndarray:
    x_start, x_stop = ensure_range(x_start, x_stop, width)
    y_start, y_stop = ensure_range(y_start, y_stop, height)
    mask = np.zeros((height, width), dtype=bool)
    mask[y_start:y_stop, x_start:x_stop] = True
    return mask


def fit_zero_loss_peak(
    energy_axis: np.ndarray,
    intensity: np.ndarray,
    fit_window: Tuple[float, float] = (-0.05, 0.05),
    guess_window: Tuple[float, float] = (-0.04, 0.04),
) -> ZeroLossFit:
    fit_mask = (energy_axis >= fit_window[0]) & (energy_axis <= fit_window[1])
    x_fit = energy_axis[fit_mask]
    y_fit = intensity[fit_mask]
    if x_fit.size < 5:
        raise ValueError("Zero-loss fit window does not contain enough points.")

    guess_mask = (x_fit >= guess_window[0]) & (x_fit <= guess_window[1])
    if guess_mask.sum() < 3:
        guess_mask = slice(None)

    background = LinearModel(prefix="bg_")
    peak = PseudoVoigtModel(prefix="p1_")
    params = background.guess(y_fit[guess_mask], x=x_fit[guess_mask])
    params += peak.guess(y_fit[guess_mask], x=x_fit[guess_mask])
    model = background + peak
    result = model.fit(y_fit, params, x=x_fit)

    center_ev = float(result.best_values["p1_center"])
    calibrated_axis = energy_axis - center_ev
    return ZeroLossFit(
        center_ev=center_ev,
        amplitude=float(result.best_values["p1_amplitude"]),
        sigma=float(result.best_values["p1_sigma"]),
        fraction=float(result.best_values["p1_fraction"]),
        background_slope=float(result.best_values["bg_slope"]),
        background_intercept=float(result.best_values["bg_intercept"]),
        fit_x=x_fit,
        fit_y=y_fit,
        best_fit=np.asarray(result.best_fit, dtype=float),
        calibrated_axis=np.asarray(calibrated_axis, dtype=float),
    )


def shift_1d_with_zeros(values: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.zeros_like(values)
    if shift == 0:
        shifted[:] = values
        return shifted
    if shift > 0:
        shifted[shift:] = values[:-shift]
    else:
        shifted[:shift] = values[-shift:]
    return shifted


def estimate_shift_1d(reference: np.ndarray, candidate: np.ndarray, max_shift: int = 64) -> int:
    reference = np.asarray(reference, dtype=float) - np.mean(reference)
    candidate = np.asarray(candidate, dtype=float) - np.mean(candidate)
    full = np.correlate(candidate, reference, mode="full")
    lags = np.arange(-reference.size + 1, reference.size)
    if max_shift is not None:
        mask = (lags >= -max_shift) & (lags <= max_shift)
        full = full[mask]
        lags = lags[mask]
    best_lag = int(lags[np.argmax(full)])
    return -best_lag


def align_spectra_1d(spectra: np.ndarray, max_shift: int = 64) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=float)
    aligned = np.zeros_like(spectra)
    reference = spectra[0]
    for index, spectrum in enumerate(spectra):
        shift = estimate_shift_1d(reference, spectrum, max_shift=max_shift)
        aligned[index] = shift_1d_with_zeros(spectrum, shift)
    return aligned


def align_spectrum_to_center(
    energy_axis: np.ndarray,
    spectrum: np.ndarray,
    center_ev: float,
) -> np.ndarray:
    shifted_axis = np.asarray(energy_axis, dtype=float) + float(center_ev)
    return np.interp(
        shifted_axis,
        np.asarray(energy_axis, dtype=float),
        np.asarray(spectrum, dtype=float),
        left=0.0,
        right=0.0,
    )


def shift_2d_with_zeros(image: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    shifted = np.zeros_like(image)

    src_y_start = max(0, -shift_y)
    src_y_stop = min(image.shape[0], image.shape[0] - shift_y) if shift_y >= 0 else image.shape[0]
    dst_y_start = max(0, shift_y)
    dst_y_stop = dst_y_start + (src_y_stop - src_y_start)

    src_x_start = max(0, -shift_x)
    src_x_stop = min(image.shape[1], image.shape[1] - shift_x) if shift_x >= 0 else image.shape[1]
    dst_x_start = max(0, shift_x)
    dst_x_stop = dst_x_start + (src_x_stop - src_x_start)

    if src_y_stop > src_y_start and src_x_stop > src_x_start:
        shifted[dst_y_start:dst_y_stop, dst_x_start:dst_x_stop] = image[src_y_start:src_y_stop, src_x_start:src_x_stop]
    return shifted


def estimate_shift_2d(reference: np.ndarray, candidate: np.ndarray, max_shift: int = 12) -> Tuple[int, int]:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    best_score = None
    best_shift = (0, 0)
    for shift_y in range(-max_shift, max_shift + 1):
        for shift_x in range(-max_shift, max_shift + 1):
            shifted = shift_2d_with_zeros(candidate, shift_y, shift_x)
            score = float(np.sum(reference * shifted))
            if best_score is None or score > best_score:
                best_score = score
                best_shift = (shift_y, shift_x)
    return best_shift


def align_stack_2d(images: np.ndarray, max_shift: int = 12) -> np.ndarray:
    images = np.asarray(images, dtype=float)
    aligned = np.zeros_like(images)
    reference = images[0]
    for index, image in enumerate(images):
        shift_y, shift_x = estimate_shift_2d(reference, image, max_shift=max_shift)
        aligned[index] = shift_2d_with_zeros(image, shift_y, shift_x)
    return aligned


def process_map_dataset(
    signal,
    *,
    energy_range: Tuple[int, int],
    polygon_mask: np.ndarray,
    intensity_range: Tuple[float, float],
    display_image: Optional[np.ndarray] = None,
    fit_window: Tuple[float, float] = (-0.05, 0.05),
    guess_window: Tuple[float, float] = (-0.04, 0.04),
) -> MapProcessingResult:
    if signal.data.ndim != 3:
        raise ValueError("Map mode expects a 3D dataset shaped like (y, x, energy).")

    data = np.asarray(signal.data)
    y_size, x_size, e_size = data.shape
    e_start, e_stop = ensure_range(energy_range[0], energy_range[1], e_size)
    intensity_image = data[:, :, e_start:e_stop].sum(axis=2)
    if polygon_mask.shape != intensity_image.shape:
        raise ValueError("Polygon mask shape does not match the map image shape.")

    z_min = min(float(intensity_range[0]), float(intensity_range[1]))
    z_max = max(float(intensity_range[0]), float(intensity_range[1]))
    threshold_mask = (intensity_image >= z_min) & (intensity_image <= z_max)
    selection_mask = np.asarray(polygon_mask, dtype=bool) & threshold_mask
    selected_spectra = data[selection_mask]
    if selected_spectra.size == 0:
        raise ValueError("No pixels passed the current polygon ROI and intensity range.")

    energy_axis = spectral_axis_from_signal(signal, selected_spectra.shape[1])
    individual_fits = [
        fit_zero_loss_peak(
            energy_axis,
            spectrum,
            fit_window=fit_window,
            guess_window=guess_window,
        )
        for spectrum in selected_spectra
    ]
    centers = np.asarray([fit.center_ev for fit in individual_fits], dtype=float)
    aligned_spectra = np.asarray(
        [
            align_spectrum_to_center(energy_axis, spectrum, center_ev)
            for spectrum, center_ev in zip(selected_spectra, centers, strict=False)
        ],
        dtype=float,
    )
    summed_spectrum = aligned_spectra.sum(axis=0)
    zlp_fit = fit_zero_loss_peak(
        energy_axis,
        summed_spectrum,
        fit_window=fit_window,
        guess_window=guess_window,
    )

    return MapProcessingResult(
        intensity_image=np.asarray(intensity_image, dtype=float),
        display_image=np.asarray(
            display_image if display_image is not None else intensity_image,
            dtype=float,
        ),
        masked_image=np.where(selection_mask, intensity_image, np.nan),
        selection_mask=selection_mask,
        selected_pixel_count=int(selection_mask.sum()),
        selected_spectra=np.asarray(aligned_spectra, dtype=float),
        selected_zlp_centers_ev=centers,
        summed_spectrum=np.asarray(summed_spectrum, dtype=float),
        energy_axis_raw=np.asarray(energy_axis, dtype=float),
        energy_axis_calibrated=zlp_fit.calibrated_axis,
        zero_loss_fit=zlp_fit,
    )


def process_snapshot_stack(
    signal,
    *,
    vertical_range: Tuple[int, int],
    frame_range: Optional[Tuple[int, int]] = None,
    fit_window: Tuple[float, float] = (-0.05, 0.05),
    guess_window: Tuple[float, float] = (-0.04, 0.04),
) -> StackProcessingResult:
    if signal.data.ndim != 3:
        raise ValueError(
            "Snapshot-stack mode expects a 3D dataset shaped like (frame, y, energy)."
        )

    data = np.asarray(signal.data)
    frames, y_size, _ = data.shape
    y_start, y_stop = ensure_range(vertical_range[0], vertical_range[1], y_size)
    if frame_range is None:
        f_start, f_stop = 0, frames
    else:
        f_start, f_stop = ensure_range(frame_range[0], frame_range[1], frames)

    detector_image_raw = data[f_start:f_stop].sum(axis=0)
    slices = []
    for index in range(f_start, f_stop):
        aligned_slice = align_spectra_1d(data[index, y_start:y_stop, :].copy())
        slices.append(np.asarray(aligned_slice, dtype=float))

    aligned_stack = np.asarray(slices, dtype=float)
    aligned_stack = align_stack_2d(aligned_stack.copy())

    summed_spectrum = aligned_stack.sum(axis=0).sum(axis=0)
    energy_axis = spectral_axis_from_signal(signal, summed_spectrum.shape[0])
    zlp_fit = fit_zero_loss_peak(
        energy_axis,
        summed_spectrum,
        fit_window=fit_window,
        guess_window=guess_window,
    )

    return StackProcessingResult(
        detector_image_raw=np.asarray(detector_image_raw, dtype=float),
        aligned_stack=aligned_stack,
        summed_spectrum=np.asarray(summed_spectrum, dtype=float),
        energy_axis_raw=np.asarray(energy_axis, dtype=float),
        energy_axis_calibrated=zlp_fit.calibrated_axis,
        zero_loss_fit=zlp_fit,
    )


def ev_to_cminv(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float) * EV_TO_CMINV
