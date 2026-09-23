from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpiceKernelConfig:
    """
    SPICE kernel configuration required by Version 3 §5.1.

    The paths point to the kernels used by the solar-geometry stage.
    """

    lsk_path: str
    pck_path: str
    spk_path: str
    fk_path: str

@dataclass(frozen=True)
class SolarGeometry:
    """
    Solar geometry for one image location and timestamp.

    azimuth is measured clockwise from selenographic north.

    elevation is the solar elevation angle above the local horizontal.

    Both angles are stored in radians.
    """

    azimuth: float
    elevation: float
def elevation_is_usable(
    elevation: float,
    minimum_degrees: float = 2.0,
    maximum_degrees: float = 80.0,
) -> bool:
    """
    Return whether solar elevation lies inside the Version 3
    usable shadow-segmentation range.

    Parameters
    ----------
    elevation:
        Solar elevation in radians.

    minimum_degrees:
        Lower usable elevation bound in degrees.

    maximum_degrees:
        Upper usable elevation bound in degrees.

    Returns
    -------
    bool
        True when the elevation is inside the usable range.
        False means the tile should be routed to case C4.
    """

    minimum = np.deg2rad(minimum_degrees)
    maximum = np.deg2rad(maximum_degrees)

    return minimum <= elevation <= maximum