from pathlib import Path

import numpy as np
import spiceypy as spice

from patchrift.geometry.solar import SpiceKernelConfig
def wrap_angle_pi(angle: float) -> float:
    """
    Wrap an angle in radians into [-pi, pi).
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

def load_spice_kernels(config: SpiceKernelConfig) -> None:
    """
    Load the SPICE kernels required by Version 3 §5.1.

    The caller is responsible for ensuring that the supplied kernel
    files exist and are compatible with the intended ephemeris setup.
    """

    for kernel_path in (
        config.lsk_path,
        config.pck_path,
        config.spk_path,
        config.fk_path,
    ):
        path = Path(kernel_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"SPICE kernel not found: {path}"
            )

        spice.furnsh(str(path))
def sun_vector_moon_me(
    timestamp: str,
) -> tuple[np.ndarray, float]:
    """
    Return the apparent Sun position relative to the Moon,
    expressed in the MOON_ME frame.

    The timestamp is interpreted by SPICE's STR2ET routine.

    Returns
    -------
    state:
        Six-element state vector. The first three elements are the
        Sun position relative to the Moon in kilometres. The final
        three elements are the corresponding velocity components.

    light_time:
        One-way light time in seconds.

    Notes
    -----
    Version 3 §5.1 requires light-time and stellar-aberration
    correction, implemented here with SPICE's LT+S option.
    """

    et = spice.str2et(timestamp)

    state, light_time = spice.spkezr(
        "SUN",
        et,
        "MOON_ME",
        "LT+S",
        "MOON",
    )

    return np.asarray(state, dtype=float), float(light_time)
def local_enu_basis(
    latitude: float,
    longitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the local East, North, Up basis in the MOON_ME frame.

    Parameters
    ----------
    latitude:
        Selenographic latitude in radians.

    longitude:
        Selenographic longitude in radians.

    Returns
    -------
    east, north, up:
        Three orthonormal unit vectors expressed in MOON_ME coordinates.

    Notes
    -----
    The MOON_ME frame has +Z along the north mean lunar rotation axis
    and its prime meridian defines the zero-longitude direction.
    """

    cos_lat = np.cos(latitude)
    sin_lat = np.sin(latitude)
    cos_lon = np.cos(longitude)
    sin_lon = np.sin(longitude)

    east = np.array(
        [-sin_lon, cos_lon, 0.0],
        dtype=float,
    )

    north = np.array(
        [
            -sin_lat * cos_lon,
            -sin_lat * sin_lon,
            cos_lat,
        ],
        dtype=float,
    )

    up = np.array(
        [
            cos_lat * cos_lon,
            cos_lat * sin_lon,
            sin_lat,
        ],
        dtype=float,
    )

    return east, north, up
def solar_azimuth_elevation(
    state: np.ndarray,
    latitude: float,
    longitude: float,
) -> tuple[float, float]:
    """
    Convert a Moon-relative Sun state into local solar azimuth
    and elevation.

    Parameters
    ----------
    state:
        Six-element Sun state returned by sun_vector_moon_me().
        The first three elements are the position vector in km.

    latitude:
        Selenographic latitude in radians.

    longitude:
        Selenographic longitude in radians.

    Returns
    -------
    azimuth:
        Solar azimuth in radians, measured clockwise from
        selenographic north.

    elevation:
        Solar elevation in radians above the local horizontal.
    """

    state = np.asarray(state, dtype=float)

    if state.shape != (6,):
        raise ValueError("state must have shape (6,)")

    position = state[:3]

    east, north, up = local_enu_basis(
        latitude,
        longitude,
    )

    east_component = np.dot(position, east)
    north_component = np.dot(position, north)
    up_component = np.dot(position, up)

    horizontal_and_vertical = np.array(
        [
            east_component,
            north_component,
            up_component,
        ],
        dtype=float,
    )

    magnitude = np.linalg.norm(horizontal_and_vertical)

    if magnitude == 0.0:
        raise ValueError("Sun position vector has zero magnitude")

    azimuth = np.arctan2(
        east_component,
        north_component,
    )

    elevation = np.arcsin(
        up_component / magnitude,
    )

    return float(azimuth), float(elevation)
def solar_geometry_at_location(
    timestamp: str,
    latitude: float,
    longitude: float,
) -> "SolarGeometry":
    """
    Compute Version 3 §5.1 solar geometry for a lunar surface location.

    Parameters
    ----------
    timestamp:
        Observation timestamp understood by SPICE.

    latitude:
        Selenographic latitude in radians.

    longitude:
        Selenographic longitude in radians.

    Returns
    -------
    SolarGeometry
        Solar azimuth and elevation in radians.
    """

    state, _ = sun_vector_moon_me(timestamp)

    azimuth, elevation = solar_azimuth_elevation(
        state,
        latitude,
        longitude,
    )

    from patchrift.geometry.solar import SolarGeometry

    return SolarGeometry(
        azimuth=azimuth,
        elevation=elevation,
    )
def solar_azimuth_derivatives(
    timestamp: str,
    latitude: float,
    longitude: float,
    latitude_step: float,
    longitude_step: float,
) -> tuple[float, float]:
    """
    Evaluate the finite-difference derivatives of solar azimuth
    with respect to selenographic latitude and longitude.

    Version 3 §5.1 specifies that these derivatives are evaluated
    by finite differences on the same ephemeris query and retained
    for the geometric uncertainty term.

    Parameters
    ----------
    timestamp:
        Observation timestamp understood by SPICE.

    latitude:
        Selenographic latitude in radians.

    longitude:
        Selenographic longitude in radians.

    latitude_step:
        Finite-difference step in radians.

    longitude_step:
        Finite-difference step in radians.

    Returns
    -------
    dA_dlatitude:
        ∂A/∂latitude.

    dA_dlongitude:
        ∂A/∂longitude.
    """

    if latitude_step <= 0.0:
        raise ValueError(
            "latitude_step must be positive"
        )

    if longitude_step <= 0.0:
        raise ValueError(
            "longitude_step must be positive"
        )

    state, _ = sun_vector_moon_me(timestamp)

    azimuth_lat_plus, _ = solar_azimuth_elevation(
        state,
        latitude + latitude_step,
        longitude,
    )

    azimuth_lat_minus, _ = solar_azimuth_elevation(
        state,
        latitude - latitude_step,
        longitude,
    )

    azimuth_lon_plus, _ = solar_azimuth_elevation(
        state,
        latitude,
        longitude + longitude_step,
    )

    azimuth_lon_minus, _ = solar_azimuth_elevation(
        state,
        latitude,
        longitude - longitude_step,
    )

    dA_dlatitude = wrap_angle_pi(
        azimuth_lat_plus - azimuth_lat_minus
    ) / (2.0 * latitude_step)

    dA_dlongitude = wrap_angle_pi(
        azimuth_lon_plus - azimuth_lon_minus
    ) / (2.0 * longitude_step)

    return (
        float(dA_dlatitude),
        float(dA_dlongitude),
    )