import numpy as np


def fit_bg_poly(x, y_obs, poly_order=5, fit_areas=None):
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y_obs, dtype=float).reshape(-1)
    n = min(x_arr.size, y_arr.size)
    if n == 0:
        return y_arr[:0]
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.count_nonzero(valid) < 2:
        return np.asarray(y_arr, dtype=float)
    if fit_areas:
        area_mask = np.zeros_like(valid, dtype=bool)
        for area in fit_areas:
            try:
                xmin = float(area[0])
                xmax = float(area[1])
            except Exception:
                continue
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            area_mask |= (x_arr >= xmin) & (x_arr <= xmax)
        fit_mask = valid & area_mask
        if np.count_nonzero(fit_mask) < 2:
            fit_mask = valid
    else:
        fit_mask = valid
    x_fit = x_arr[fit_mask]
    y_fit = y_arr[fit_mask]
    deg = max(0, min(int(poly_order), x_fit.size - 1))
    coeffs = np.polyfit(x_fit, y_fit, deg)
    y_bg = np.polyval(coeffs, x_arr)
    return np.asarray(y_bg, dtype=float)
