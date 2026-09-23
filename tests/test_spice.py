import numpy as np
import spiceypy as spice

from patchrift.geometry.spice import (
    load_spice_kernels,
    sun_vector_moon_me,
)
from patchrift.geometry.solar import SpiceKernelConfig


def test_sun_vector_moon_me():
    config = SpiceKernelConfig(
        lsk_path="kernels/spice/naif0012.tls",
        pck_path="kernels/spice/moon_pa_de440_200625.bpc",
        spk_path="kernels/spice/de440.bsp",
        fk_path="kernels/spice/moon_de440_250416.tf",
    )

    load_spice_kernels(config)

    try:
        state, light_time = sun_vector_moon_me(
            "2026 SEP 21 00:00:00 UTC"
        )
        
        assert state.shape == (6,)
        assert np.all(np.isfinite(state))
        assert np.isfinite(light_time)
        assert light_time > 0.0
    finally:
        spice.kclear()