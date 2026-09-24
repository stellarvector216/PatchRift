import numpy as np


def use_full_image_formulation(image_pixel_count, patch_size, keypoint_count):
    """Return V3 §9's switch decision: K P² > N/2."""
    values = (image_pixel_count, patch_size, keypoint_count)
    if any(not isinstance(value, (int, np.integer)) for value in values):
        raise ValueError("pixel count, patch size, and keypoint count must be integers")
    if image_pixel_count <= 0 or patch_size <= 0 or keypoint_count < 0:
        raise ValueError("pixel count and patch size must be positive; keypoints nonnegative")
    return keypoint_count * patch_size**2 > image_pixel_count / 2.0

def calculate_section9_complexity(image_dim, P, K, num_scales=4, num_orientations=12):
    """
    Implements Section 9's theoretical FFT-operation and storage estimates.
    ``speed_multiplier`` is an operation-count ratio, not measured wall-clock
    runtime speedup.
    Compares the O(N log N) full-image FFT constraint against the O(K * P^2 log P^2) patch architecture.
    
    image_dim: Dimension of the square input image (e.g., 5000 for 5000x5000 pixels)
    P: Patch size (e.g., 128)
    K: Number of extracted keypoints/patches (e.g., 1000)
    """
    values = (image_dim, P, num_scales, num_orientations)
    if any(not isinstance(value, (int, np.integer)) or value <= 0 for value in values):
        raise ValueError("image_dim, P, num_scales, and num_orientations must be positive integers")
    if not isinstance(K, (int, np.integer)) or K < 0:
        raise ValueError("K must be a nonnegative integer")

    # N is the total number of pixels in the raw image
    N = image_dim ** 2
    
    # Area of a single canonical patch
    P_area = P ** 2
    
    # 1. Computational Operations (Big-O notation for FFT)
    # A 2D FFT requires approximately 5 * N * log2(N) floating point operations
    ops_full_image = 5 * N * np.log2(N)
    
    # Patch-based executes K individual small FFTs
    ops_patch_based = K * (5 * P_area * np.log2(P_area))
    
    # 2. Memory Complexity
    # Phase congruency requires storing 64-bit complex float arrays (16 bytes per pixel)
    # across all scales and orientations simultaneously.
    num_filters = num_scales * num_orientations
    
    mem_full_image_mb = (N * num_filters * 16) / (1024 ** 2)
    
    # The patch-based pipeline reuses memory sequentially, so peak memory 
    # is constrained to just one patch at a time.
    mem_patch_based_mb = (P_area * num_filters * 16) / (1024 ** 2) if K else 0.0
    
    speed_multiplier = float("inf") if K == 0 else ops_full_image / ops_patch_based
    
    return speed_multiplier, mem_full_image_mb, mem_patch_based_mb
