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

        if np.any(self.wavelengths <= 0):
            raise ValueError(
                "wavelengths must be positive"
            )

        if np.any(self.snr < 0):
            raise ValueError(
                "snr cannot be negative"
            )