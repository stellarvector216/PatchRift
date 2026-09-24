from dataclasses import dataclass

import numpy as np


@dataclass
class IIRSProduct:
    """
    Raw IIRS hyperspectral product before Version 3 §4.2 band reduction.

    cube has shape:
        (number_of_bands, rows, columns)

    wavelengths contains the center wavelength of each band, in micrometres.

    bad_bands is True for bands that must be excluded before reduction.

    snr contains the per-band signal-to-noise ratio and is used as the
    fallback spectral weight when no TMC spectral-response curve is available.
    """

    cube: np.ndarray
    wavelengths: np.ndarray
    bad_bands: np.ndarray
    snr: np.ndarray
    valid_mask: np.ndarray | None = None
    band_bounds: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.cube = np.asarray(self.cube, dtype=np.float32)
        self.wavelengths = np.asarray(self.wavelengths, dtype=float)
        self.bad_bands = np.asarray(self.bad_bands, dtype=bool)
        self.snr = np.asarray(self.snr, dtype=float)

        if self.cube.ndim != 3:
            raise ValueError(
                "cube must have shape (bands, rows, columns)"
            )

        n_bands = self.cube.shape[0]

        if self.wavelengths.shape != (n_bands,):
            raise ValueError(
                "wavelengths must contain one value per band"
            )

        if self.bad_bands.shape != (n_bands,):
            raise ValueError(
                "bad_bands must contain one value per band"
            )

        if self.snr.shape != (n_bands,):
            raise ValueError(
                "snr must contain one value per band"
            )

        if self.band_bounds is not None:
            self.band_bounds = np.asarray(self.band_bounds, dtype=float)
            if self.band_bounds.shape != (n_bands, 2):
                raise ValueError("band_bounds must have shape (bands, 2)")
            if (
                not np.all(np.isfinite(self.band_bounds))
                or np.any(self.band_bounds[:, 0] >= self.band_bounds[:, 1])
            ):
                raise ValueError("band_bounds must contain finite lower/upper wavelength pairs")
            if np.any(
                (self.wavelengths < self.band_bounds[:, 0])
                | (self.wavelengths > self.band_bounds[:, 1])
            ):
                raise ValueError("each wavelength centre must lie inside its band bounds")

        if np.any(self.wavelengths <= 0):
            raise ValueError(
                "wavelengths must be positive"
            )

        if not np.all(np.isfinite(self.snr)) or np.any(self.snr < 0):
            raise ValueError(
                "snr must be finite and non-negative"
            )

        if self.valid_mask is None:
            self.valid_mask = np.isfinite(self.cube)
        else:
            self.valid_mask = np.asarray(self.valid_mask, dtype=bool)

            if self.valid_mask.shape != self.cube.shape:
                raise ValueError(
                    "valid_mask must have the same shape as cube"
                )

        if np.any(self.valid_mask & ~np.isfinite(self.cube)):
            raise ValueError(
                "valid IIRS pixels must contain only finite values"
            )
