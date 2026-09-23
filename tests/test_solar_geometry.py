import numpy as np
from unittest.mock import patch

from patchrift.geometry.spice import (
    solar_azimuth_derivatives,
    solar_azimuth_elevation,
    solar_geometry_at_location,
)
from patchrift.geometry.solar import elevation_is_usable

def test_solar_azimuth_elevation_sun_due_east():
    state = np.array(
        [
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    azimuth, elevation = solar_azimuth_elevation(
        state,
        latitude=0.0,
        longitude=0.0,
    )

    np.testing.assert_allclose(
        azimuth,
        np.pi / 2,
    )
    np.testing.assert_allclose(elevation, 0.0)


def test_solar_azimuth_elevation_sun_due_north():
    state = np.array(
        [
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    azimuth, elevation = solar_azimuth_elevation(
        state,
        latitude=0.0,
        longitude=0.0,
    )

    np.testing.assert_allclose(azimuth, 0.0)
    np.testing.assert_allclose(elevation, 0.0)


def test_solar_azimuth_elevation_sun_at_45_degree_elevation():
    state = np.array(
        [
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    azimuth, elevation = solar_azimuth_elevation(
        state,
        latitude=0.0,
        longitude=0.0,
    )

    np.testing.assert_allclose(
        azimuth,
        0.0,
    )
    np.testing.assert_allclose(
        elevation,
        np.pi / 4,
    )
    from unittest.mock import patch


def test_solar_geometry_at_location_composes_spice_and_projection():
    fake_state = np.array(
        [
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    with patch(
        "patchrift.geometry.spice.sun_vector_moon_me",
        return_value=(fake_state, 500.0),
    ):
        geometry = solar_geometry_at_location(
            timestamp="2000-01-01T00:00:00",
            latitude=0.0,
            longitude=0.0,
        )

    np.testing.assert_allclose(
        geometry.azimuth,
        np.pi / 2,
    )
    np.testing.assert_allclose(
        geometry.elevation,
        0.0,
    )
def test_solar_azimuth_derivatives_use_finite_difference():
    fake_state = np.array(
        [
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    with patch(
        "patchrift.geometry.spice.sun_vector_moon_me",
        return_value=(fake_state, 500.0),
    ):
        dA_dlatitude, dA_dlongitude = (
            solar_azimuth_derivatives(
                timestamp="2000-01-01T00:00:00",
                latitude=0.0,
                longitude=0.0,
                latitude_step=1e-5,
                longitude_step=1e-5,
            )
        )

    assert np.isfinite(dA_dlatitude)
    assert np.isfinite(dA_dlongitude)
def test_elevation_is_usable_inside_range():
    assert elevation_is_usable(
        np.deg2rad(20.0)
    )


def test_elevation_is_usable_accepts_lower_boundary():
    assert elevation_is_usable(
        np.deg2rad(2.0)
    )


def test_elevation_is_usable_accepts_upper_boundary():
    assert elevation_is_usable(
        np.deg2rad(80.0)
    )


def test_elevation_is_usable_rejects_below_lower_boundary():
    assert not elevation_is_usable(
        np.deg2rad(1.9)
    )


def test_elevation_is_usable_rejects_above_upper_boundary():
    assert not elevation_is_usable(
        np.deg2rad(80.1)
    )