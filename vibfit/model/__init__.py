"""Data model for vibfit (spectrum, sections, fits)."""

from .fitting import build_fit, clone_region, default_region, fit_background, load_spectrum
from .session_io import export_fit_results, export_plot_npy, get_param_dir, list_backup_events, load_session_from_backup, save_session, update_backup_comment
from .state import (
    BackgroundArea,
    BackgroundFitResult,
    EV_TO_CMINV,
    BackgroundSpec,
    FitRegion,
    FitResultBundle,
    NOTEBOOK_PRESETS,
    ParameterConstraint,
    PeakResult,
    PeakSpec,
    SavedSection,
    SpectrumData,
    cminv_to_ev,
    ev_to_cminv,
)

__all__ = [
    "BackgroundArea",
    "BackgroundFitResult",
    "BackgroundSpec",
    "EV_TO_CMINV",
    "FitRegion",
    "FitResultBundle",
    "NOTEBOOK_PRESETS",
    "ParameterConstraint",
    "PeakResult",
    "PeakSpec",
    "SavedSection",
    "SpectrumData",
    "build_fit",
    "clone_region",
    "cminv_to_ev",
    "default_region",
    "ev_to_cminv",
    "export_fit_results",
    "export_plot_npy",
    "fit_background",
    "load_spectrum",
    "get_param_dir",
    "list_backup_events",
    "load_session_from_backup",
    "save_session",
    "update_backup_comment",
]
