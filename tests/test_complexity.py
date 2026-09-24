import numpy as np
import pytest

from patchrift.utils.complexity import calculate_section9_complexity, use_full_image_formulation


def test_complexity_reports_theoretical_ratio_and_zero_keypoint_case():
    speed, full_mb, patch_mb = calculate_section9_complexity(12000, 128, 500)
    assert speed > 30.0
    assert full_mb > 100000.0
    assert patch_mb < 15.0
    zero_speed, _, zero_patch = calculate_section9_complexity(12000, 128, 0)
    assert np.isinf(zero_speed)
    assert zero_patch == 0.0


def test_complexity_formulation_switch_uses_strict_v3_threshold():
    assert not use_full_image_formulation(10000, 10, 50)
    assert use_full_image_formulation(10000, 10, 51)
    assert not use_full_image_formulation(10000, 10, 0)


@pytest.mark.parametrize(
    "args",
    [
        (0, 128, 4),
        (12000, 0, 4),
        (12000, 128, -1),
        (12000, 128, 4, 0, 12),
        (12000, 128, 4, 4, 0),
    ],
)
def test_complexity_rejects_invalid_dimensions_and_counts(args):
    with pytest.raises(ValueError):
        calculate_section9_complexity(*args)
