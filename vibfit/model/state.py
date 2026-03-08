from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


EV_TO_CMINV = 8065.54429


def ev_to_cminv(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * EV_TO_CMINV


def cminv_to_ev(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) / EV_TO_CMINV


@dataclass
class SpectrumData:
    path: str
    x_ev: np.ndarray
    intensity: np.ndarray
    source_kind: str
    title: str = ""

    @property
    def x_cminv(self) -> np.ndarray:
        return np.asarray(ev_to_cminv(self.x_ev), dtype=float)


@dataclass
class ParameterConstraint:
    value: float
    vary: bool = True
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class PeakSpec:
    name: str
    guess_min_cminv: float
    guess_max_cminv: float
    amplitude: ParameterConstraint
    center: ParameterConstraint
    sigma: ParameterConstraint
    fraction: ParameterConstraint


@dataclass
class BackgroundSpec:
    model_name: str = "PowerLaw"
    anchor_left_cminv: Optional[float] = None
    anchor_right_cminv: Optional[float] = None
    fit_areas: list["BackgroundArea"] = field(default_factory=list)


@dataclass
class BackgroundArea:
    x_min_cminv: float
    x_max_cminv: float


@dataclass
class FitRegion:
    name: str
    x_min_cminv: float
    x_max_cminv: float
    background: BackgroundSpec
    peaks: list[PeakSpec] = field(default_factory=list)


@dataclass
class BackgroundFitResult:
    x_cminv: np.ndarray
    y_raw: np.ndarray
    background: np.ndarray
    y_bgsub: np.ndarray
    bgsub_offset: float
    area_mask: np.ndarray
    fit_report: str
    success: bool
    chisqr: float
    redchi: float
    aic: float
    bic: float


@dataclass
class PeakResult:
    name: str
    center_ev: float
    center_cminv: float
    amplitude: float
    sigma_ev: float
    sigma_cminv: float
    fraction: float
    curve: np.ndarray


@dataclass
class FitResultBundle:
    region_name: str
    x_cminv: np.ndarray
    y_raw: np.ndarray
    best_fit: np.ndarray
    background: np.ndarray
    y_bgsub: np.ndarray
    bgsub_offset: float
    best_fit_bgsub: np.ndarray
    residual_raw: np.ndarray
    residual_bgsub: np.ndarray
    peaks: list[PeakResult]
    fit_report: str
    success: bool
    chisqr: float
    redchi: float
    aic: float
    bic: float


@dataclass
class SavedSection:
    timestamp: str
    label: str
    region: FitRegion
    background_result: Optional[BackgroundFitResult] = None
    fit_result: Optional[FitResultBundle] = None


def _peak(
    name: str,
    guess_window_cm: tuple[float, float],
    center_window_cm: tuple[float, float],
    *,
    amp_min: Optional[float] = None,
    sigma_max_cm: Optional[float] = None,
    fraction_max: Optional[float] = None,
) -> PeakSpec:
    sigma_value_cm = max((center_window_cm[1] - center_window_cm[0]) / 8.0, 20.0)
    return PeakSpec(
        name=name,
        guess_min_cminv=float(guess_window_cm[0]),
        guess_max_cminv=float(guess_window_cm[1]),
        amplitude=ParameterConstraint(value=100.0, vary=True, min=amp_min),
        center=ParameterConstraint(
            value=float((center_window_cm[0] + center_window_cm[1]) / 2.0),
            vary=True,
            min=float(center_window_cm[0]),
            max=float(center_window_cm[1]),
        ),
        sigma=ParameterConstraint(
            value=float(sigma_value_cm),
            vary=True,
            min=0.0,
            max=None if sigma_max_cm is None else float(sigma_max_cm),
        ),
        fraction=ParameterConstraint(
            value=0.5,
            vary=True,
            min=0.0,
            max=fraction_max,
        ),
    )


NOTEBOOK_PRESETS: list[FitRegion] = [
    FitRegion(
        name="300-1400 cm^-1",
        x_min_cminv=300.0,
        x_max_cminv=1400.0,
        background=BackgroundSpec(
            model_name="PowerLaw",
            anchor_left_cminv=400.0,
            anchor_right_cminv=1350.0,
            fit_areas=[
                BackgroundArea(300.0, 420.0),
                BackgroundArea(1280.0, 1400.0),
            ],
        ),
        peaks=[
            _peak("p1", (450.0, 500.0), (450.0, 500.0), amp_min=0.0),
            _peak("p2", (750.0, 850.0), (750.0, 850.0)),
            _peak("p3", (1100.0, 1150.0), (1100.0, 1150.0)),
            _peak("p4", (1150.0, 1200.0), (1150.0, 1200.0)),
        ],
    ),
    FitRegion(
        name="1470-2100 cm^-1",
        x_min_cminv=1470.0,
        x_max_cminv=2100.0,
        background=BackgroundSpec(
            model_name="PowerLaw",
            anchor_left_cminv=1500.0,
            anchor_right_cminv=2050.0,
            fit_areas=[
                BackgroundArea(1470.0, 1560.0),
                BackgroundArea(2010.0, 2100.0),
            ],
        ),
        peaks=[
            _peak("p1", (1550.0, 1700.0), (1600.0, 1650.0), amp_min=0.0, fraction_max=0.1),
            _peak("p2", (1700.0, 1900.0), (1800.0, 1900.0), sigma_max_cm=70.0),
            _peak("p3", (1900.0, 1950.0), (1940.0, 1970.0), amp_min=5.0, sigma_max_cm=100.0, fraction_max=1.0),
        ],
    ),
    FitRegion(
        name="2100-2500 cm^-1",
        x_min_cminv=2100.0,
        x_max_cminv=2500.0,
        background=BackgroundSpec(
            model_name="PowerLaw",
            anchor_left_cminv=2130.0,
            anchor_right_cminv=2450.0,
            fit_areas=[
                BackgroundArea(2100.0, 2160.0),
                BackgroundArea(2420.0, 2500.0),
            ],
        ),
        peaks=[
            _peak("p4", (2200.0, 2300.0), (2200.0, 2300.0), sigma_max_cm=70.0, fraction_max=0.1),
            _peak("p5", (2250.0, 2350.0), (2250.0, 2300.0), sigma_max_cm=70.0, fraction_max=0.1),
        ],
    ),
    FitRegion(
        name="2500-4500 cm^-1",
        x_min_cminv=2500.0,
        x_max_cminv=4500.0,
        background=BackgroundSpec(
            model_name="PowerLaw",
            anchor_left_cminv=3000.0,
            anchor_right_cminv=3800.0,
            fit_areas=[
                BackgroundArea(2500.0, 2900.0),
                BackgroundArea(3900.0, 4500.0),
            ],
        ),
        peaks=[
            _peak("peak1", (3200.0, 3300.0), (3200.0, 3300.0), sigma_max_cm=200.0, fraction_max=1.0),
            _peak("peak2", (3400.0, 3500.0), (3450.0, 3550.0), sigma_max_cm=200.0, fraction_max=1.0),
            _peak("peak3", (3500.0, 3600.0), (3450.0, 3650.0), amp_min=1.0, sigma_max_cm=200.0, fraction_max=1.0),
        ],
    ),
]
