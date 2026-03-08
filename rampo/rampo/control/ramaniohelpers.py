import os
import glob
import numpy as np

from ..utils import get_temp_dir, readchi
from ..ds_ramspec import SpectrumPattern


def load_spectrum_xy(spectrum_path, spectrum_cache):
    if spectrum_path in spectrum_cache:
        return spectrum_cache[spectrum_path]
    ext = os.path.splitext(str(spectrum_path))[1].lower()
    if ext == ".chi":
        __, __, x, y = readchi(spectrum_path)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    else:
        spectrum = SpectrumPattern()
        spectrum.read_file(spectrum_path)
        x, y = spectrum.get_raw()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    spectrum_cache[spectrum_path] = (x, y)
    return x, y


def load_bgsub_or_raw_xy(spectrum_path, use_bgsub, spectrum_cache):
    if not bool(use_bgsub):
        return load_spectrum_xy(spectrum_path, spectrum_cache)

    # Priority 1: temp bgsub file under <data>-rampo/
    try:
        temp_dir = get_temp_dir(spectrum_path)
        base = os.path.splitext(os.path.basename(spectrum_path))[0]
        temp_bgsub = os.path.join(temp_dir, f"{base}.bgsub.chi")
        if os.path.exists(temp_bgsub):
            __, __, x, y = readchi(temp_bgsub)
            return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    except Exception:
        pass

    # Priority 2: sibling bgsub file next to chi
    sibling_bgsub = os.path.splitext(spectrum_path)[0] + ".bgsub.chi"
    if os.path.exists(sibling_bgsub):
        __, __, x, y = readchi(sibling_bgsub)
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    # Fallback: raw chi if bgsub file is unavailable.
    return load_spectrum_xy(spectrum_path, spectrum_cache)


def find_temp_cake_triplet(chi_path):
    temp_dir = get_temp_dir(chi_path)
    tth_files = sorted(glob.glob(os.path.join(temp_dir, "*.tth.cake.npy")))
    if not tth_files:
        return None

    stem_map = {}
    for tth_f in tth_files:
        stem = tth_f[: -len(".tth.cake.npy")]
        azi_f = stem + ".azi.cake.npy"
        int_f = stem + ".int.cake.npy"
        if os.path.exists(azi_f) and os.path.exists(int_f):
            stem_map[stem] = (tth_f, azi_f, int_f)
    if not stem_map:
        return None

    # Most recent triplet first.
    triplets = sorted(stem_map.values(), key=lambda t: os.path.getmtime(t[2]), reverse=True)
    return triplets[0]


def load_cake_data(chi_path, cake_cache):
    if chi_path in cake_cache:
        return cake_cache[chi_path]

    triplet = find_temp_cake_triplet(chi_path)
    if triplet is None:
        return None

    tth = np.load(triplet[0])
    azi = np.load(triplet[1])
    intensity = np.load(triplet[2])
    payload = (
        np.asarray(tth, dtype=float),
        np.asarray(azi, dtype=float),
        np.asarray(intensity, dtype=float),
    )
    cake_cache[chi_path] = payload
    return payload


load_chi_xy = load_spectrum_xy
