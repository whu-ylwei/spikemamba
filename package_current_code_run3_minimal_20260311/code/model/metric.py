import numpy as np


def abs_rel_diff(y_input, y_target, eps = 1e-6):
    abs_diff = np.abs(y_target-y_input)
    return (abs_diff[~np.isnan(abs_diff)]/(y_target[~np.isnan(y_target)]+eps)).mean()

def squ_rel_diff(y_input, y_target, eps = 1e-6):
    abs_diff = np.abs(y_target-y_input)
    is_nan = np.isnan(abs_diff)
    return (abs_diff[~is_nan]**2/(y_target[~is_nan]**2+eps)).mean()

def rms_linear(y_input, y_target):
    abs_diff = np.abs(y_target-y_input)
    is_nan = np.isnan(abs_diff)
    return np.sqrt((abs_diff[~is_nan]**2).mean())

def scale_invariant_error(y_input, y_target):
    log_diff = np.abs(y_target-y_input)
    is_nan = np.isnan(log_diff)
    return (log_diff[~is_nan]**2).mean()-(log_diff[~is_nan].mean())**2

def mean_error(y_input, y_target):
    abs_diff = np.abs(y_target-y_input)
    return abs_diff[~np.isnan(abs_diff)].mean()

def median_error(y_input, y_target):
    abs_diff = np.abs(y_target-y_input)
    return np.median(abs_diff[~np.isnan(abs_diff)])

def mse(y_input, y_target):
    if y_input.shape != y_target.shape:
        raise ValueError(f"Shape mismatch: {y_input.shape} vs {y_target.shape}")

    diff = y_input - y_target
    valid_mask = ~np.isnan(y_target)
    if not np.any(valid_mask):
        return 0.0

    sq = diff[valid_mask] ** 2
    return float(np.mean(sq))


def structural_similarity(y_input, y_target):
    raise NotImplementedError(
        "structural_similarity metric requires optional image-metric dependencies."
    )
