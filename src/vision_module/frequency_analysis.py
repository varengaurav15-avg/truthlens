import math
import numpy as np
from scipy import fft


def high_frequency_energy(img: np.ndarray) -> float:
    """Fraction of FFT magnitude in high-frequency components. Returns float in [0, 1]."""
    if img.ndim == 3:
        img = np.mean(img, axis=2).astype(np.float32)
    else:
        img = img.astype(np.float32)

    f = fft.fft2(img)
    fshift = fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = fshift.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4

    Y, X = np.ogrid[:h, :w]
    mask_low = (Y - cy) ** 2 + (X - cx) ** 2 <= r ** 2

    total = magnitude.sum() + 1e-9
    high = magnitude[~mask_low].sum()
    return float(high / total)


def frequency_fake_score(img: np.ndarray) -> float:
    """Convert high-frequency energy into a P(fake) proxy in [0, 1]. Use as a soft signal."""
    hfe = high_frequency_energy(img)
    # Sigmoid centred at 0.84 with steepness 20
    score = 1.0 / (1.0 + math.exp(-20.0 * (hfe - 0.84)))
    return round(float(score), 4)
