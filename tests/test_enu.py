import numpy as np

from patchrift.geometry.spice import local_enu_basis


def test_local_enu_basis_at_equator_prime_meridian():
    east, north, up = local_enu_basis(
        latitude=0.0,
        longitude=0.0,
    )

    np.testing.assert_allclose(
        east,
        [0.0, 1.0, 0.0],
    )

    np.testing.assert_allclose(
        north,
        [0.0, 0.0, 1.0],
    )

    np.testing.assert_allclose(
        up,
        [1.0, 0.0, 0.0],
    )


def test_local_enu_basis_is_orthonormal():
    east, north, up = local_enu_basis(
        latitude=0.37,
        longitude=-1.21,
    )

    np.testing.assert_allclose(np.linalg.norm(east), 1.0)
    np.testing.assert_allclose(np.linalg.norm(north), 1.0)
    np.testing.assert_allclose(np.linalg.norm(up), 1.0)

    np.testing.assert_allclose(np.dot(east, north), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(east, up), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(north, up), 0.0, atol=1e-12)