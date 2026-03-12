from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from typing import Optional
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from .state import BackgroundFitResult, FitRegion, FitResultBundle, SavedSection, SpectrumData


MANIFEST_FILE = "vibfit_manifest.json"
SESSION_FILE = "vibfit_session.json"
BACKUP_INDEX_FILE = "vibfit_backup_index.json"


@dataclass
class SaveResult:
    param_dir: str
    manifest_path: str
    backup_id: Optional[str]


@dataclass
class ExportResult:
    json_path: str
    excel_path: str


@dataclass
class NpyExportResult:
    npy_path: str
    script_path: str
    png_path: str
    pdf_path: str


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _atomic_write_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=_json_default).encode("utf-8")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=os.path.dirname(path)) as handle:
            tmp_path = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _state_payload(
    spectrum: SpectrumData | None,
    region: FitRegion,
    background_result: BackgroundFitResult | None,
    fit_result: FitResultBundle | None,
    saved_sections: list[SavedSection],
):
    return {
        "spectrum_path": None if spectrum is None else spectrum.path,
        "region": asdict(region),
        "background_result": None if background_result is None else asdict(background_result),
        "fit_result": None if fit_result is None else asdict(fit_result),
        "saved_sections": [asdict(section) for section in saved_sections],
    }


def _state_hash(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_index(param_dir: str):
    return _load_json(
        os.path.join(param_dir, BACKUP_INDEX_FILE),
        {"format_family": "vibfit-session", "format_version": 1, "next_index": 1, "events": [], "latest_backup_id": None},
    )


def _snapshot_dir(param_dir: str, backup_id: str) -> str:
    return os.path.join(param_dir, backup_id)


def get_param_dir(base_path: str) -> str:
    directory = os.path.dirname(base_path)
    stem = os.path.splitext(os.path.basename(base_path))[0]
    param_dir = os.path.join(directory, f"{stem}-vibfit")
    os.makedirs(param_dir, exist_ok=True)
    return param_dir


def save_session(
    base_path: str,
    spectrum: SpectrumData | None,
    region: FitRegion,
    background_result: BackgroundFitResult | None,
    fit_result: FitResultBundle | None,
    saved_sections: list[SavedSection],
    reason: str = "save",
) -> SaveResult:
    param_dir = get_param_dir(base_path)
    payload = _state_payload(spectrum, region, background_result, fit_result, saved_sections)
    index = _load_index(param_dir)
    new_hash = _state_hash(payload)

    latest_id = index.get("latest_backup_id")
    if latest_id:
        latest_payload = _load_json(os.path.join(_snapshot_dir(param_dir, latest_id), SESSION_FILE), None)
        if latest_payload is not None and _state_hash(latest_payload) == new_hash:
            return SaveResult(param_dir=param_dir, manifest_path=os.path.join(_snapshot_dir(param_dir, latest_id), MANIFEST_FILE), backup_id=latest_id)

    backup_id = str(int(index.get("next_index", 0)))
    index["next_index"] = int(index.get("next_index", 1)) + 1
    snap_dir = _snapshot_dir(param_dir, backup_id)
    manifest = {
        "format_family": "vibfit-session",
        "format_version": 1,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "files": {"session": SESSION_FILE, "backup_index": os.path.join("..", BACKUP_INDEX_FILE)},
    }
    _atomic_write_json(os.path.join(snap_dir, SESSION_FILE), payload)
    _atomic_write_json(os.path.join(snap_dir, MANIFEST_FILE), manifest)
    index["latest_backup_id"] = backup_id
    index["events"].append(
        {
            "id": backup_id,
            "timestamp": manifest["created_at"],
            "reason": reason,
            "summary": region.name,
            "state_hash": new_hash,
        }
    )
    _atomic_write_json(os.path.join(param_dir, BACKUP_INDEX_FILE), index)
    return SaveResult(param_dir=param_dir, manifest_path=os.path.join(snap_dir, MANIFEST_FILE), backup_id=backup_id)


def list_backup_events(param_dir: str):
    index = _load_index(param_dir)
    return list(index.get("events", []))


def update_backup_comment(param_dir: str, event_id: str, comment: str) -> bool:
    index = _load_index(param_dir)
    events = index.get("events", [])
    target_id = str(event_id)
    for event in events:
        if str(event.get("id", "")) == target_id:
            event["reason"] = str(comment)
            _atomic_write_json(os.path.join(param_dir, BACKUP_INDEX_FILE), index)
            return True
    return False


def load_session_from_backup(param_dir: str, event_id: str | None = None):
    index = _load_index(param_dir)
    target = event_id or index.get("latest_backup_id")
    if not target:
        return None
    session_path = os.path.join(_snapshot_dir(param_dir, target), SESSION_FILE)
    if not os.path.exists(session_path):
        return None
    return _load_json(session_path, None)


def _xlsx_col_name(index: int) -> str:
    result = ""
    index += 1
    while index > 0:
        index, rem = divmod(index - 1, 26)
        result = chr(65 + rem) + result
    return result


def _xlsx_cell_ref(row_index: int, col_index: int) -> str:
    return f"{_xlsx_col_name(col_index)}{row_index + 1}"


def _xlsx_value_xml(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return f'<c t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, int) and not isinstance(value, bool):
        return f"<c><v>{value}</v></c>"
    if isinstance(value, float):
        return f"<c><v>{value}</v></c>"
    text = xml_escape(str(value))
    return f'<c t="inlineStr"><is><t>{text}</t></is></c>'


def _xlsx_sheet_xml(rows) -> str:
    row_xml = []
    for row_index, row in enumerate(rows):
        cells = []
        for col_index, value in enumerate(row):
            if value is None or value == "":
                continue
            cell_xml = _xlsx_value_xml(value)
            cell_xml = cell_xml.replace("<c", f'<c r="{_xlsx_cell_ref(row_index, col_index)}"', 1)
            cells.append(cell_xml)
        row_xml.append(f'<row r="{row_index + 1}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        "</worksheet>"
    )


def _write_xlsx(path: str, sheets: list[tuple[str, list[list[object]]]]):
    workbook_sheets = []
    workbook_rels = []
    content_overrides = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, (name, rows) in enumerate(sheets, start=1):
            sheet_path = f"xl/worksheets/sheet{idx}.xml"
            zf.writestr(sheet_path, _xlsx_sheet_xml(rows))
            workbook_sheets.append(
                f'<sheet name="{xml_escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
            )
            workbook_rels.append(
                f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
            )
            content_overrides.append(
                f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        zf.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
                f'{"".join(content_overrides)}'
                "</Types>"
            ),
        )
        zf.writestr(
            "_rels/.rels",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
                "</Relationships>"
            ),
        )
        zf.writestr(
            "xl/workbook.xml",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f'<sheets>{"".join(workbook_sheets)}</sheets>'
                "</workbook>"
            ),
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f'{"".join(workbook_rels)}'
                "</Relationships>"
            ),
        )
        created = dt.datetime.now().isoformat(timespec="seconds")
        zf.writestr(
            "docProps/core.xml",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/" '
                'xmlns:dcterms="http://purl.org/dc/terms/" '
                'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
                '<dc:creator>vibfit</dc:creator>'
                '<cp:lastModifiedBy>vibfit</cp:lastModifiedBy>'
                f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>'
                f'<dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>'
                "</cp:coreProperties>"
            ),
        )
        zf.writestr(
            "docProps/app.xml",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
                'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
                '<Application>vibfit</Application>'
                "</Properties>"
            ),
        )


def _peak_setup_rows(region: FitRegion) -> list[list[object]]:
    setup_headers = [
        "name",
        "guess_min_cminv",
        "guess_max_cminv",
        "amplitude",
        "amplitude_min",
        "amplitude_max",
        "amplitude_vary",
        "center_cminv",
        "center_min",
        "center_max",
        "center_vary",
        "sigma_cminv",
        "sigma_min",
        "sigma_max",
        "sigma_vary",
        "fraction",
        "fraction_min",
        "fraction_max",
        "fraction_vary",
    ]
    rows = [setup_headers]
    for peak in region.peaks:
        rows.append(
            [
                peak.name,
                peak.guess_min_cminv,
                peak.guess_max_cminv,
                peak.amplitude.value,
                peak.amplitude.min,
                peak.amplitude.max,
                int(peak.amplitude.vary),
                peak.center.value,
                peak.center.min,
                peak.center.max,
                int(peak.center.vary),
                peak.sigma.value,
                peak.sigma.min,
                peak.sigma.max,
                int(peak.sigma.vary),
                peak.fraction.value,
                peak.fraction.min,
                peak.fraction.max,
                int(peak.fraction.vary),
            ]
        )
    return rows


def _peak_result_rows(fit_result: FitResultBundle) -> list[list[object]]:
    result_headers = [
        "name",
        "center_cminv",
        "center_ev",
        "amplitude",
        "sigma_cminv",
        "sigma_ev",
        "fraction",
    ]
    rows = [result_headers]
    for peak in fit_result.peaks:
        rows.append(
            [
                peak.name,
                peak.center_cminv,
                peak.center_ev,
                peak.amplitude,
                peak.sigma_cminv,
                peak.sigma_ev,
                peak.fraction,
            ]
        )
    return rows


def _curve_rows(background_result: BackgroundFitResult, fit_result: FitResultBundle) -> list[list[object]]:
    headers = [
        "fit_x_cminv",
        "fit_y_raw",
        "fit_background",
        "fit_y_bgsub",
        "best_fit",
        "best_fit_bgsub",
        "residual_raw",
        "residual_bgsub",
    ]
    headers.extend(f"peak_{peak.name}" for peak in fit_result.peaks)
    rows = [headers]
    for idx, values in enumerate(
        zip(
            fit_result.x_cminv,
            fit_result.y_raw,
            fit_result.background,
            fit_result.y_bgsub,
            fit_result.best_fit,
            fit_result.best_fit_bgsub,
            fit_result.residual_raw,
            fit_result.residual_bgsub,
            strict=False,
        )
    ):
        row = [float(value) for value in values]
        for peak in fit_result.peaks:
            curve = np.asarray(peak.curve, dtype=float)
            row.append(float(curve[idx]) if idx < curve.size else None)
        rows.append(row)

    rows.append([])
    rows.append(["background_x_cminv", "background_y_raw", "background_curve", "background_y_bgsub", "background_mask"])
    for values in zip(
        background_result.x_cminv,
        background_result.y_raw,
        background_result.background,
        background_result.y_bgsub,
        background_result.area_mask,
        strict=False,
    ):
        rows.append(
            [
                float(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
                int(values[4]),
            ]
        )
    return rows


def _section_sheet_name(index: int, section: SavedSection, used_names: set[str]) -> str:
    base = (section.label or section.region.name or f"Section {index}").strip() or f"Section {index}"
    safe = "".join("_" if ch in '[]:*?/\\' else ch for ch in base)
    safe = safe[:31] or f"Section {index}"
    candidate = safe
    suffix = 1
    while candidate in used_names:
        tail = f"_{suffix}"
        candidate = f"{safe[: max(0, 31 - len(tail))]}{tail}" or f"Section_{index}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _section_sheet_rows(section: SavedSection, spectrum: SpectrumData) -> list[list[object]]:
    if section.background_result is None or section.fit_result is None:
        raise ValueError("Saved sections must include both background and peak-fit results for XLS export.")

    region = section.region
    background_result = section.background_result
    fit_result = section.fit_result
    xbg_values = np.asarray(background_result.x_cminv, dtype=float)
    xpfit_values = np.asarray(fit_result.x_cminv, dtype=float)
    xbg_min = float(np.nanmin(xbg_values)) if xbg_values.size else None
    xbg_max = float(np.nanmax(xbg_values)) if xbg_values.size else None
    xpfit_min = float(np.nanmin(xpfit_values)) if xpfit_values.size else None
    xpfit_max = float(np.nanmax(xpfit_values)) if xpfit_values.size else None

    rows: list[list[object]] = [
        ["Saved", section.timestamp],
        ["Comment", section.label],
        ["Spectrum path", spectrum.path],
        ["Spectrum title", spectrum.title],
        ["Source kind", spectrum.source_kind],
        ["Region name", region.name],
        ["xbg min", xbg_min],
        ["xbg max", xbg_max],
        ["xpfit min", xpfit_min],
        ["xpfit max", xpfit_max],
        ["Background success", int(background_result.success)],
        ["Background redchi", background_result.redchi],
        ["Fit success", int(fit_result.success)],
        ["Fit chisqr", fit_result.chisqr],
        ["Fit redchi", fit_result.redchi],
        ["Fit aic", fit_result.aic],
        ["Fit bic", fit_result.bic],
        [],
        ["Peak setup"],
    ]
    rows.extend(_peak_setup_rows(region))
    rows.append([])
    rows.append(["Peak results"])
    rows.extend(_peak_result_rows(fit_result))
    rows.append([])
    rows.append(["Curves"])
    rows.extend(_curve_rows(background_result, fit_result))
    return rows


def export_fit_results(
    output_base: str,
    spectrum: SpectrumData,
    region: FitRegion,
    background_result: BackgroundFitResult,
    fit_result: FitResultBundle,
) -> ExportResult:
    json_path = f"{output_base}.json"
    excel_path = f"{output_base}.xlsx"
    payload = {
        "format_family": "vibfit-fit-export",
        "format_version": 1,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "spectrum": {
            "path": spectrum.path,
            "title": spectrum.title,
            "source_kind": spectrum.source_kind,
        },
        "region": asdict(region),
        "background_fit": asdict(background_result),
        "peak_fit": asdict(fit_result),
    }
    _atomic_write_json(json_path, payload)

    summary_rows = [
        ("Spectrum path", spectrum.path),
        ("Spectrum title", spectrum.title),
        ("Source kind", spectrum.source_kind),
        ("Region", fit_result.region_name),
        ("Background success", int(background_result.success)),
        ("Background redchi", background_result.redchi),
        ("Fit success", int(fit_result.success)),
        ("Fit chisqr", fit_result.chisqr),
        ("Fit redchi", fit_result.redchi),
        ("Fit aic", fit_result.aic),
        ("Fit bic", fit_result.bic),
    ]
    setup_headers = [
        "name",
        "guess_min_cminv",
        "guess_max_cminv",
        "amplitude",
        "amplitude_min",
        "amplitude_max",
        "amplitude_vary",
        "center_cminv",
        "center_min",
        "center_max",
        "center_vary",
        "sigma_cminv",
        "sigma_min",
        "sigma_max",
        "sigma_vary",
        "fraction",
        "fraction_min",
        "fraction_max",
        "fraction_vary",
    ]
    setup_rows = [setup_headers]
    for row, peak in enumerate(region.peaks, start=1):
        values = [
            peak.name,
            peak.guess_min_cminv,
            peak.guess_max_cminv,
            peak.amplitude.value,
            peak.amplitude.min,
            peak.amplitude.max,
            int(peak.amplitude.vary),
            peak.center.value,
            peak.center.min,
            peak.center.max,
            int(peak.center.vary),
            peak.sigma.value,
            peak.sigma.min,
            peak.sigma.max,
            int(peak.sigma.vary),
            peak.fraction.value,
            peak.fraction.min,
            peak.fraction.max,
            int(peak.fraction.vary),
        ]
        setup_rows.append(values)

    result_headers = [
        "name",
        "center_cminv",
        "center_ev",
        "amplitude",
        "sigma_cminv",
        "sigma_ev",
        "fraction",
    ]
    result_rows = [result_headers]
    for row, peak in enumerate(fit_result.peaks, start=1):
        values = [
            peak.name,
            peak.center_cminv,
            peak.center_ev,
            peak.amplitude,
            peak.sigma_cminv,
            peak.sigma_ev,
            peak.fraction,
        ]
        result_rows.append(values)

    curve_headers = ["x_cminv", "y_raw", "background", "y_bgsub", "best_fit", "best_fit_bgsub", "residual_raw", "residual_bgsub"]
    curve_rows = [curve_headers]
    for row, values in enumerate(
        zip(
            fit_result.x_cminv,
            fit_result.y_raw,
            fit_result.background,
            fit_result.y_bgsub,
            fit_result.best_fit,
            fit_result.best_fit_bgsub,
            fit_result.residual_raw,
            fit_result.residual_bgsub,
            strict=False,
        ),
        start=1,
    ):
        curve_rows.append([float(value) for value in values])

    _write_xlsx(
        excel_path,
        [
            ("summary", [[label, value] for label, value in summary_rows]),
            ("setup", setup_rows),
            ("results", result_rows),
            ("curves", curve_rows),
        ],
    )
    return ExportResult(json_path=json_path, excel_path=excel_path)


def export_saved_sections(
    output_base: str,
    spectrum: SpectrumData,
    saved_sections: list[SavedSection],
) -> ExportResult:
    if not saved_sections:
        raise ValueError("No saved sections available for export.")

    export_sections = []
    sheets: list[tuple[str, list[list[object]]]] = []
    used_sheet_names: set[str] = set()
    for index, section in enumerate(saved_sections, start=1):
        if section.background_result is None or section.fit_result is None:
            continue
        export_sections.append(asdict(section))
        sheet_name = _section_sheet_name(index, section, used_sheet_names)
        sheets.append((sheet_name, _section_sheet_rows(section, spectrum)))

    if not sheets:
        raise ValueError("Saved sections do not contain complete background and peak-fit results.")

    json_path = f"{output_base}.json"
    excel_path = f"{output_base}.xlsx"
    payload = {
        "format_family": "vibfit-sections-export",
        "format_version": 1,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "spectrum": {
            "path": spectrum.path,
            "title": spectrum.title,
            "source_kind": spectrum.source_kind,
        },
        "saved_sections": export_sections,
    }
    _atomic_write_json(json_path, payload)
    _write_xlsx(excel_path, sheets)
    return ExportResult(json_path=json_path, excel_path=excel_path)


def export_plot_npy(
    output_base: str,
    spectrum: SpectrumData,
    region: FitRegion,
    background_result: BackgroundFitResult,
    fit_result: FitResultBundle,
    view_limits: dict[str, tuple[float, float]] | None = None,
) -> NpyExportResult:
    output_dir = os.path.dirname(output_base)
    npy_path = os.path.join(output_dir, "vibfit-data.npy")
    script_path = os.path.join(output_dir, "vibfit-script.py")
    png_path = os.path.join(output_dir, "vibfit-plot.png")
    pdf_path = os.path.join(output_dir, "vibfit-plot.pdf")
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    payload = {
        "region_name": fit_result.region_name,
        "x_cminv": np.asarray(fit_result.x_cminv, dtype=float),
        "y_raw": np.asarray(fit_result.y_raw, dtype=float),
        "background": np.asarray(fit_result.background, dtype=float),
        "best_fit": np.asarray(fit_result.best_fit, dtype=float),
        "y_bgsub": np.asarray(fit_result.y_bgsub, dtype=float),
        "best_fit_bgsub": np.asarray(fit_result.best_fit_bgsub, dtype=float),
        "residual_raw": np.asarray(fit_result.residual_raw, dtype=float),
        "residual_bgsub": np.asarray(fit_result.residual_bgsub, dtype=float),
        "region_x_min": float(region.x_min_cminv),
        "region_x_max": float(region.x_max_cminv),
        "bg_areas": [
            (float(area.x_min_cminv), float(area.x_max_cminv))
            for area in region.background.fit_areas
        ],
        "peaks": [
            {
                "name": peak.name,
                "center_cminv": float(peak.center_cminv),
                "curve": np.asarray(peak.curve, dtype=float),
            }
            for peak in fit_result.peaks
        ],
        "spectrum_path": spectrum.path,
        "spectrum_title": spectrum.title,
        "source_kind": spectrum.source_kind,
        "background_redchi": float(background_result.redchi),
        "fit_redchi": float(fit_result.redchi),
        "view_limits": {} if view_limits is None else {
            key: (float(value[0]), float(value[1]))
            for key, value in view_limits.items()
        },
    }
    np.save(npy_path, payload, allow_pickle=True)
    script = f"""import numpy as np
import matplotlib.pyplot as plt

data = np.load(r"{npy_path}", allow_pickle=True).item()
x = np.asarray(data["x_cminv"], dtype=float)
y_raw = np.asarray(data["y_raw"], dtype=float)
background = np.asarray(data["background"], dtype=float)
best_fit = np.asarray(data["best_fit"], dtype=float)
y_bgsub = np.asarray(data["y_bgsub"], dtype=float)
best_fit_bgsub = np.asarray(data["best_fit_bgsub"], dtype=float)
residual_raw = np.asarray(data["residual_raw"], dtype=float)
residual_bgsub = np.asarray(data["residual_bgsub"], dtype=float)
peaks = data["peaks"]
view_limits = data.get("view_limits", {{}})

fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
fig.patch.set_facecolor("#2b2d31")
for ax in (ax_top, ax_bottom):
    ax.set_facecolor("#1e1f22")
    ax.grid(True, color="#43464d", alpha=0.35, linewidth=0.6)
    ax.tick_params(colors="#f0f0f0")
    ax.xaxis.label.set_color("#f0f0f0")
    ax.yaxis.label.set_color("#f0f0f0")
    for spine in ax.spines.values():
        spine.set_color("#c9c9c9")

ax_top.plot(x, y_raw, color="#f2f2f2", linewidth=1.0, label="vibEELS data")
ax_top.axvspan(float(data["region_x_min"]), float(data["region_x_max"]), color="#3b82f6", alpha=0.12, label="Fit window")
for i, (x0, x1) in enumerate(data["bg_areas"]):
    ax_top.axvspan(min(x0, x1), max(x0, x1), color="#f59e0b", alpha=0.12, label="Bg fit area" if i == 0 else None)
ax_top.plot(x, background, color="#60a5fa", linewidth=1.0, label="background")
ax_top.plot(x, best_fit, color="#ff4d6d", linewidth=1.0, label="best fit")
raw_span = float(np.nanmax(y_raw) - np.nanmin(y_raw)) if y_raw.size else 0.0
residual_span = float(np.nanmax(residual_raw) - np.nanmin(residual_raw)) if residual_raw.size else 0.0
top_zero = float(np.nanmin(y_raw) - max(0.06 * max(raw_span, 1.0), 1.5 * residual_span))
ax_top.axhline(top_zero, color="#a1a1aa", linewidth=0.8, linestyle="--", label="shifted zero")
ax_top.fill_between(x, top_zero, residual_raw + top_zero, color="#e5e7eb", alpha=0.28, label="residue")
ax_top.plot(x, residual_raw + top_zero, color="#e5e7eb", linewidth=0.9)
ax_top.set_ylabel("Raw intensity")
ax_top.legend(loc="upper right")
if "raw_xlim" in view_limits:
    ax_top.set_xlim(*view_limits["raw_xlim"])
if "raw_ylim" in view_limits:
    ax_top.set_ylim(*view_limits["raw_ylim"])

ax_bottom.plot(x, y_bgsub, color="#f2f2f2", linewidth=1.0)
ax_bottom.plot(x, best_fit_bgsub, color="#ff4d6d", linewidth=1.0, label="best fit")
for peak in peaks:
    ax_bottom.plot(x, np.asarray(peak["curve"], dtype=float), linewidth=1.0, label=f'{{peak["center_cminv"]:.0f}} cm$^{{-1}}$')
fit_span = float(np.nanmax(y_bgsub) - np.nanmin(y_bgsub)) if y_bgsub.size else 0.0
fit_resid_span = float(np.nanmax(residual_bgsub) - np.nanmin(residual_bgsub)) if residual_bgsub.size else 0.0
bottom_zero = float(np.nanmin(y_bgsub) - max(0.08 * max(fit_span, 1.0), 1.5 * fit_resid_span))
ax_bottom.axhline(bottom_zero, color="#a1a1aa", linewidth=0.8, linestyle="--")
ax_bottom.fill_between(x, bottom_zero, residual_bgsub + bottom_zero, color="#93c5fd", alpha=0.25, label="residue")
ax_bottom.plot(x, residual_bgsub + bottom_zero, color="#93c5fd", linewidth=0.9)
ax_bottom.set_ylabel("Bg-sub intensity")
ax_bottom.set_xlabel("Wavenumber (cm$^{{-1}}$)")
ax_bottom.legend(loc="upper right")
if "fit_xlim" in view_limits:
    ax_bottom.set_xlim(*view_limits["fit_xlim"])
if "fit_ylim" in view_limits:
    ax_bottom.set_ylim(*view_limits["fit_ylim"])

plt.show()
"""
    with open(script_path, "w", encoding="utf-8") as handle:
        handle.write(script)
    _render_export_plot(payload, png_path, pdf_path)
    return NpyExportResult(
        npy_path=npy_path,
        script_path=script_path,
        png_path=png_path,
        pdf_path=pdf_path,
    )


def _render_export_plot(payload, png_path: str, pdf_path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(payload["x_cminv"], dtype=float)
    y_raw = np.asarray(payload["y_raw"], dtype=float)
    background = np.asarray(payload["background"], dtype=float)
    best_fit = np.asarray(payload["best_fit"], dtype=float)
    y_bgsub = np.asarray(payload["y_bgsub"], dtype=float)
    best_fit_bgsub = np.asarray(payload["best_fit_bgsub"], dtype=float)
    residual_raw = np.asarray(payload["residual_raw"], dtype=float)
    residual_bgsub = np.asarray(payload["residual_bgsub"], dtype=float)
    peaks = payload["peaks"]
    view_limits = payload.get("view_limits", {})

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#2b2d31")
    for ax in (ax_top, ax_bottom):
        ax.set_facecolor("#1e1f22")
        ax.grid(True, color="#43464d", alpha=0.35, linewidth=0.6)
        ax.tick_params(colors="#f0f0f0")
        ax.xaxis.label.set_color("#f0f0f0")
        ax.yaxis.label.set_color("#f0f0f0")
        for spine in ax.spines.values():
            spine.set_color("#c9c9c9")

    ax_top.plot(x, y_raw, color="#f2f2f2", linewidth=1.0, label="vibEELS data")
    ax_top.axvspan(float(payload["region_x_min"]), float(payload["region_x_max"]), color="#3b82f6", alpha=0.12, label="Fit window")
    for i, (x0, x1) in enumerate(payload["bg_areas"]):
        ax_top.axvspan(min(x0, x1), max(x0, x1), color="#f59e0b", alpha=0.12, label="Bg fit area" if i == 0 else None)
    ax_top.plot(x, background, color="#60a5fa", linewidth=1.0, label="background")
    ax_top.plot(x, best_fit, color="#ff4d6d", linewidth=1.0, label="best fit")
    raw_span = float(np.nanmax(y_raw) - np.nanmin(y_raw)) if y_raw.size else 0.0
    residual_span = float(np.nanmax(residual_raw) - np.nanmin(residual_raw)) if residual_raw.size else 0.0
    top_zero = float(np.nanmin(y_raw) - max(0.06 * max(raw_span, 1.0), 1.5 * residual_span))
    ax_top.axhline(top_zero, color="#a1a1aa", linewidth=0.8, linestyle="--", label="shifted zero")
    ax_top.fill_between(x, top_zero, residual_raw + top_zero, color="#e5e7eb", alpha=0.28, label="residue")
    ax_top.plot(x, residual_raw + top_zero, color="#e5e7eb", linewidth=0.9)
    ax_top.set_ylabel("Raw intensity")
    ax_top.legend(loc="upper right")
    if "raw_xlim" in view_limits:
        ax_top.set_xlim(*view_limits["raw_xlim"])
    if "raw_ylim" in view_limits:
        ax_top.set_ylim(*view_limits["raw_ylim"])

    ax_bottom.plot(x, y_bgsub, color="#f2f2f2", linewidth=1.0)
    ax_bottom.plot(x, best_fit_bgsub, color="#ff4d6d", linewidth=1.0, label="best fit")
    for peak in peaks:
        ax_bottom.plot(x, np.asarray(peak["curve"], dtype=float), linewidth=1.0, label=f'{peak["center_cminv"]:.0f} cm$^{{-1}}$')
    fit_span = float(np.nanmax(y_bgsub) - np.nanmin(y_bgsub)) if y_bgsub.size else 0.0
    fit_resid_span = float(np.nanmax(residual_bgsub) - np.nanmin(residual_bgsub)) if residual_bgsub.size else 0.0
    bottom_zero = float(np.nanmin(y_bgsub) - max(0.08 * max(fit_span, 1.0), 1.5 * fit_resid_span))
    ax_bottom.axhline(bottom_zero, color="#a1a1aa", linewidth=0.8, linestyle="--")
    ax_bottom.fill_between(x, bottom_zero, residual_bgsub + bottom_zero, color="#93c5fd", alpha=0.25, label="residue")
    ax_bottom.plot(x, residual_bgsub + bottom_zero, color="#93c5fd", linewidth=0.9)
    ax_bottom.set_ylabel("Bg-sub intensity")
    ax_bottom.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax_bottom.legend(loc="upper right")
    if "fit_xlim" in view_limits:
        ax_bottom.set_xlim(*view_limits["fit_xlim"])
    if "fit_ylim" in view_limits:
        ax_bottom.set_ylim(*view_limits["fit_ylim"])

    fig.savefig(png_path, dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
