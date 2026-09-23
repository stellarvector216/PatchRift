import numpy as np
import pytest

from patchrift.geometry.solar import (
    SolarGeometry,
    SpiceKernelConfig,
)
from patchrift.geometry.spice import load_spice_kernels

def test_spice_kernel_config_stores_kernel_paths():
    config = SpiceKernelConfig(
        lsk_path="kernels/naif0012.tls",
        pck_path="kernels/pck00011.tpc",
        spk_path="kernels/de440.bsp",
        fk_path="kernels/moon_de440_250416.tf",
    )

    assert config.lsk_path == "kernels/naif0012.tls"
    assert config.pck_path == "kernels/pck00011.tpc"
    assert config.spk_path == "kernels/de440.bsp"
    

def test_solar_geometry_stores_radian_angles():
    geometry = SolarGeometry(
        azimuth=np.pi / 2,
        elevation=np.pi / 4,
    )

    assert geometry.azimuth == np.pi / 2
    assert geometry.elevation == np.pi / 4
def test_load_spice_kernels_rejects_missing_kernel():
    config = SpiceKernelConfig(
        lsk_path="missing/naif0012.tls",
        pck_path="missing/pck00011.tpc",
        spk_path="missing/de440.bsp",
        fk_path="missing/moon_de440_250416.tf",
    )

    with pytest.raises(
        FileNotFoundError,
        match="SPICE kernel not found",
    ):
        load_spice_kernels(config)