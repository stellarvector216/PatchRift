import numpy as np
import pytest

from patchrift.cascade.stage_iii import stage3_geometric_conditioning
from patchrift.features.stage_iv import (
    calculate_moment_analysis,
    principal_curvature_ratio_test,
    validate_downsample_factor
)

def test_stage3_diagnostics():
    # 1. Setup a controlled synthetic terrain (100x100 grid)
    # A single bright "impulse" pixel in the center to track filter geometry
    image = np.ones((100, 100), dtype=float)
    image[50, 50] = 100.0 
    
    gsd_current = 1.0 # meters/pixel
    gsd_target = 2.0  # meters/pixel -> sigma = 2.0
    
    # Setup dummy masks
    valid_mask = np.ones((100, 100), dtype=bool)
    masks = {
        'S_int': np.zeros((100, 100), dtype=bool) # No shadows for this test
    }
    
    # Solar azimuth at 45 degrees (pi/4)
    phi_pix = np.pi / 4 
    lambda_max = 3.0 # pixels
    
    # Execute the function
    I_cond, scaled_masks, pre_blurred = stage3_geometric_conditioning(
        image, gsd_current, gsd_target, masks, phi_pix, lambda_max, valid_mask
    )
    
    # ==========================================
    # TEST 1: Decimation Scale & Blur Geometry
    # ==========================================
    assert I_cond.shape == (50, 50), f"Scale failure: Expected (50, 50), got {I_cond.shape}"
    assert pre_blurred.shape == (100, 100), "Pre-blurred image lost its native resolution!"

    # ==========================================
    # TEST 2: Photometric Clamp Range
    # ==========================================
    assert np.max(I_cond) <= 4.0, f"Clamp failure: Max value {np.max(I_cond)} exceeds 4.0"
    assert np.min(I_cond) >= 0.0, f"Clamp failure: Min value {np.min(I_cond)} below 0.0"

    # ==========================================
    # TEST 3: Mask Resampling Integrity 
    # ==========================================
    unique_vals = np.unique(scaled_masks['S_int'])
    assert np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [0, 1]), "Masks suffered gradient blur!"


def test_stage4_moment_analysis_and_pruning():
    P = 50
    
    # ---------------------------------------------------------
    # TEST 1: Equations 55 & 56 (Cornerness vs Edgeness)
    # ---------------------------------------------------------
    pc_edge = [np.zeros((P, P)) for _ in range(12)]
    pc_edge[0][:, P//2] = 1.0  
    
    M_max_edge, M_min_edge = calculate_moment_analysis(pc_edge)
    
    assert np.max(M_max_edge) > 0.0, "FAIL: Equation 56 missed the edgeness."
    assert np.max(M_min_edge) < 1e-5, "FAIL: Equation 56 falsely detected a corner on a straight line."
    
    pc_corner = [np.zeros((P, P)) for _ in range(12)]
    pc_corner[0][:, P//2] = 1.0  
    pc_corner[6][P//2, :] = 1.0  
    
    M_max_corner, M_min_corner = calculate_moment_analysis(pc_corner)
    
    assert M_min_corner[P//2, P//2] > 0.0, "FAIL: Equation 56 failed to detect the orthogonal corner."

    # ---------------------------------------------------------
    # TEST 2: Equation 58 (Principal Curvature Ratio Test)
    # ---------------------------------------------------------
    m_min_synthetic = np.zeros((P, P))
    m_min_synthetic[25, 25] = 10.0      
    m_min_synthetic[10:40, 10] = 5.0    
    
    valid_mask = principal_curvature_ratio_test(m_min_synthetic, r=10)
    
    assert valid_mask[25, 25] == True, "FAIL: Equation 58 wrongly rejected a sharp, valid corner."
    assert valid_mask[20, 10] == False, "FAIL: Equation 58 failed to reject the elongated straight edge."

    # ---------------------------------------------------------
    # TEST 3: Equation 57 (Downsample Survival Bound)
    # ---------------------------------------------------------
    D_min = 20.0
    lambda_min = 3.0
    
    assert validate_downsample_factor(D_min, lambda_min, d_proposed=1.5) == True
    
    with pytest.raises(ValueError):
        validate_downsample_factor(D_min, lambda_min, d_proposed=2.0)
from patchrift.features.stage_iv import calculate_patch_size_bounds

def test_stage4_equations_59_and_60():
    # Setup using exact parameters from Section 10 of the PDF
    m = 4                  # DFT bins per passband
    B_omega = 2.0          # Bandwidth in octaves
    r_taper = 0.25         # Tukey taper ratio
    r_max = 48             # Outermost pooling radius
    
    # Calculate lambda_max: minimum wavelength (3) * multiplier (2.0) ^ (scales - 1)
    lambda_max = 3.0 * (2.0 ** 3)  # 24.0
    
    P_final, P_spectral, P_spatial = calculate_patch_size_bounds(
        lambda_max=lambda_max, r_max=r_max, m=m, B_omega=B_omega, r_taper=r_taper
    )
    
    # 1. Verify Equation 59 (Spectral Bound)
    # P >= (4 * 24) / 1.5 = 64
    assert P_spectral == 64.0, f"FAIL: Eq 59 calculated {P_spectral}, expected 64.0"
    
    # 2. Verify Equation 60 (Spatial Bound)
    # P >= (2 * 48) / (1 - 0.25) = 128
    assert P_spatial == 128.0, f"FAIL: Eq 60 calculated {P_spatial}, expected 128.0"
    
    # 3. Verify Final 5-Smooth Rounding
    # max(64, 128) is 128. Since 128 is 2^7, it is already 5-smooth.
    # The Section 10 parameter table explicitly mandates P = 128.
    assert P_final == 128, f"FAIL: Final patch size {P_final} does not match expected 128"
from patchrift.features.stage_iv import (
    compute_sampling_matrix, 
    generate_tukey_window, 
    extract_canonical_patch,
    extract_canonical_mask
)
def test_stage4_equations_61_to_63():
    # ---------------------------------------------------------
    # TEST 1: Equation 61 (Sampling Matrix)
    # ---------------------------------------------------------
    sigma_X = 2.0
    alpha_X = 0.0  # North is perfectly aligned
    
    # Equation 61: M_X = sigma_X * R(alpha_X + pi/2)
    M_X = compute_sampling_matrix(sigma_X, alpha_X)
    
    # With alpha = 0, theta = pi/2. 
    # R(pi/2) maps [1, 0] to [0, 1] and [0, 1] to [-1, 0]
    expected_M = np.array([
        [0.0, -2.0],
        [2.0,  0.0]
    ])
    assert np.allclose(M_X, expected_M, atol=1e-7), "FAIL: Equation 61 matrix M_X is incorrect."

    # ---------------------------------------------------------
    # TEST 2: Equation 63 (Tukey Window)
    # ---------------------------------------------------------
    P = 128
    r_taper = 0.25
    w = generate_tukey_window(P, r_taper)
    
    # Dimension check
    assert w.shape == (P, P), "FAIL: Tukey window must match patch size PxP."
    
    # Flat top check: The center must be strictly 1.0 per Eq 63
    assert w[P//2, P//2] == 1.0, "FAIL: Tukey window center must be 1.0."
    
    # Taper check: The extreme corners must taper to near 0.0
    assert w[0, 0] < 0.01, "FAIL: Tukey window failed to taper at the edges."

    # ---------------------------------------------------------
    # TEST 3: Equation 62 (Patch and Mask Extraction)
    # ---------------------------------------------------------
    # Create a dummy conditioned image (500x500)
    I_cond = np.ones((500, 500))
    dummy_mask = np.zeros((500, 500), dtype=bool)
    dummy_mask[240:260, 240:260] = True # Create a boolean square in the center
    
    # Extract around the center
    x_k, y_k = 250, 250
    
    patch = extract_canonical_patch(I_cond, x_k, y_k, M_X, P, r_taper)
    patch_mask = extract_canonical_mask(dummy_mask, x_k, y_k, M_X, P)
    
    # Dimension check
    assert patch.shape == (P, P), "FAIL: Extracted patch must be PxP."
    assert patch_mask.shape == (P, P), "FAIL: Extracted mask must be PxP."
    
    # Mask integrity check (nearest-neighbor must preserve boolean values as mandated)
    unique_vals = np.unique(patch_mask)
    assert np.array_equal(unique_vals, [False, True]) or np.array_equal(unique_vals, [False]), "FAIL: Mask suffered gradient blur during extraction." 
from patchrift.features.stage_iv import (
    generate_log_polar_layout,
    compute_maximum_index_map,
    extract_descriptor_and_weights
)

def test_stage4_section_76():
    P = 128
    N_o = 12
    
    # ---------------------------------------------------------
    # TEST 1: Log-Polar Layout
    # ---------------------------------------------------------
    cell_map = generate_log_polar_layout(P, r1=19, r2=35, rmax=48)
    
    unique_cells = np.unique(cell_map)
    assert len(unique_cells) == 18, f"FAIL: Expected 18 values (17 cells + background), got {len(unique_cells)}"
    assert np.max(unique_cells) == 16, "FAIL: Maximum cell index must be 16."
    assert cell_map[P//2, P//2] == 0, "FAIL: Center pixel must belong to central disc (cell 0)."
    
    # ---------------------------------------------------------
    # TEST 2: Equation 64 (MIM)
    # ---------------------------------------------------------
    pc_list = [np.zeros((P, P)) for _ in range(N_o)]
    pc_list[5][10:20, 10:20] = 1.0 # Simulate orientation 5 dominating a region
    
    MIM = compute_maximum_index_map(pc_list)
    assert MIM[15, 15] == 5, "FAIL: Eq 64 failed to assign the correct dominant orientation."
    # ---------------------------------------------------------
    # TEST 3: Equations 65 & 66 (Descriptor and Weights)
    # ---------------------------------------------------------
    # Dummy amplitude sum stack filled with constant structural energy
    A_sum_stack = np.full((N_o, P, P), 2.0)
    
    R_mask = np.zeros((P, P), dtype=bool)
    S_int_mask = np.zeros((P, P), dtype=bool)
    
    # Intentionally occlude cell 0 completely with a shadow interior
    S_int_mask[cell_map == 0] = True
    
    # Intentionally occlude exactly half (rounded down) of cell 1 with a moving terminator
    cell_1_coords = np.argwhere(cell_map == 1)
    half_idx = len(cell_1_coords) // 2
    for idx in range(half_idx):
        y, x = cell_1_coords[idx]
        R_mask[y, x] = True
        
    descriptor, weights = extract_descriptor_and_weights(
        MIM, A_sum_stack, cell_map, R_mask, S_int_mask, N_o=12, v_min=0.2, clamp_val=0.2
    )
    
    # Eq 65: Dimensionality check
    assert len(descriptor) == 204, f"FAIL: Descriptor must be exactly 204 dimensions, got {len(descriptor)}"
    
    # Eq 65: Per-cell normalization check
    # Cell 2 is unoccluded, its sub-vector norm must strictly equal 1.0
    cell_2_vector = descriptor[2*N_o : 3*N_o]
    assert np.isclose(np.linalg.norm(cell_2_vector), 1.0), "FAIL: Cell 2 vector was not L2-normalized."
    
    # Eq 66: Reliability weights check
    # Cell 0 is fully covered by S_int -> must collapse to v_min (0.2)
    assert np.isclose(weights[0], 0.2), f"FAIL: Weight for fully shadowed cell must be 0.2, got {weights[0]}"
    
    # Cell 1 is partially covered by R -> calculate the exact mathematical fraction
    expected_weight_1 = 1.0 - (half_idx / len(cell_1_coords))
    assert np.isclose(weights[1], expected_weight_1), f"FAIL: Weight for terminator overlap must be {expected_weight_1}, got {weights[1]}"
    
    # Cell 2 is completely clean -> 1.0
    assert np.isclose(weights[2], 1.0), f"FAIL: Weight for clean cell must be 1.0, got {weights[2]}"
from patchrift.matching.stage_v import (
    calculate_search_window,
    roll_descriptor,
    compute_weighted_distance,
    map_to_canonical_frame,
    calculate_hough_bin_size,
    fit_huber_homography,
    compute_output_transformation,self_diagnose_alignment,
        calculate_fallback_search_window
    
)

def test_stage5_section_81_equations_67_68():
    N_o = 12
    J = 17
    
    # Eq 68: Residual Rotation Search Window
    sigma = 0.2
    window = calculate_search_window(sigma, N_o)
    assert np.array_equal(window, [-2, -1, 0, 1, 2]), f"FAIL: Expected window [-2, -1, 0, 1, 2], got {window}"
    
    # Cyclic Shift Integrity
    dummy_desc = np.zeros(J * N_o)
    dummy_desc[0] = 1.0        
    dummy_desc[1 * N_o + 5] = 1.0 
    
    rolled = roll_descriptor(dummy_desc, 2, J, N_o)
    assert rolled[2] == 1.0, "FAIL: Orientation shift failed in Cell 0."
    assert rolled[0] == 0.0, "FAIL: Old orientation was not cleared."
    assert rolled[1 * N_o + 7] == 1.0, "FAIL: Orientation shift failed in Cell 1."
    
    # Eq 67: Weighted Distance Metric
    D_A = np.zeros(J * N_o)
    D_B = np.zeros(J * N_o)
    D_A[0:N_o] = 0.5
    D_A[N_o:2*N_o] = 1.0
    
    v_A = np.ones(J)
    v_B = np.ones(J)
    v_B[1] = 0.2 
    
    dist = compute_weighted_distance(D_A, D_B, v_A, v_B, J, N_o)
    expected_dist = 5.4 / 16.2
    assert np.isclose(dist, expected_dist), f"FAIL: Expected weighted distance {expected_dist}, got {dist}"

def test_stage5_section_82_83_equations_69_70_72():
    # Eq 69: Canonical Frame Inversion
    M_X = np.array([[0.0, -2.0], [2.0, 0.0]])
    native_x, native_y = 10.0, 20.0
    canonical_q_x, canonical_q_y = map_to_canonical_frame(native_x, native_y, M_X)
    
    q_vec = np.array([canonical_q_x, canonical_q_y])
    x_reconstructed = M_X @ q_vec
    assert np.isclose(x_reconstructed[0], native_x), "FAIL: Eq 69 mapping failed X-axis inversion."
    assert np.isclose(x_reconstructed[1], native_y), "FAIL: Eq 69 mapping failed Y-axis inversion."

    # Eq 70: Hough Bin Threshold Sizing
    tau = calculate_hough_bin_size(h_max=100.0, e_A=np.deg2rad(15.0), e_B=np.deg2rad(5.0), g_star=0.25)
    expected_tau = (100.0 * np.abs(np.tan(np.deg2rad(15.0)) - np.tan(np.deg2rad(5.0)))) / 0.25
    assert np.isclose(tau, expected_tau), f"FAIL: Eq 70 yielded {tau}, expected {expected_tau}"
    
    tau_flat = calculate_hough_bin_size(0.0, np.deg2rad(15.0), np.deg2rad(5.0), 0.25)
    assert tau_flat == 3.0, f"FAIL: Eq 70 floor must be 3.0, got {tau_flat}"

    # Eq 72: Output Transformation
    M_B = np.array([[1.0, 0.0], [0.0, 1.0]])
    x_B, y_B = compute_output_transformation(x_A=100.0, y_A=100.0, t_x=5.0, t_y=-5.0, 
                                             g_A=1.0, g_B=2.0, delta_alpha=np.pi/2, M_B=M_B)
    assert np.isclose(x_B, -45.0), f"FAIL: Eq 72 X-transform expected -45.0, got {x_B}"
    assert np.isclose(y_B, 45.0), f"FAIL: Eq 72 Y-transform expected 45.0, got {y_B}"

def test_stage5_equation_71_huber_homography():
    # True transformation: Scale 2, Translate 10
    H_true = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 10.0], [0.0, 0.0, 1.0]])
    
    # Create a robust Hough peak of 25 perfect inlier points
    x_A_list = []
    for i in range(5):
        for j in range(5):
            x_A_list.append([i * 10.0, j * 10.0])
    x_A = np.array(x_A_list)
    
    x_B = np.zeros_like(x_A)
    for i in range(25):
        vec = H_true @ np.array([x_A[i, 0], x_A[i, 1], 1.0])
        x_B[i] = vec[:2] / vec[2]
        
    # Inject the realistic terrain-relief distortion into just one point
    x_B[24] += np.array([15.0, -15.0])
    
    # Init guess (Translate 10, Scale 1)
    H_init = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]])
    
    H_opt = fit_huber_homography(x_A, x_B, H_init)
    
    # The Huber loss combined with a realistic inlier count will now lock onto 2.0
    # The Huber loss locks onto sub-pixel accuracy.
    # A ~0.12 pixel translation shift from a 15-pixel outlier proves the robust fitting works.
    assert np.isclose(H_opt[0, 0], 2.0, atol=0.05), f"FAIL: Eq 71 failed to recover scale. Got {H_opt[0, 0]}"
    assert np.isclose(H_opt[1, 1], 2.0, atol=0.05), f"FAIL: Eq 71 failed to recover scale. Got {H_opt[1, 1]}"
    assert np.isclose(H_opt[0, 2], 10.0, atol=0.2), f"FAIL: Eq 71 failed to preserve translation. Got {H_opt[0, 2]}"

def test_stage5_section_84_case_c4_fallback():
    # ---------------------------------------------------------
    # TEST 1: Section 8.4 Diagnosis Trigger
    # ---------------------------------------------------------
    # Scenario A: Robust Hough peak of 25 inliers (Healthy Alignment)
    assert self_diagnose_alignment(25, N_min=10) is True, "FAIL: Sec 8.4 rejected a healthy alignment."
    # Scenario B: Collapsed Hough peak of 4 inliers (Pathological Shadows -> C4)
    assert self_diagnose_alignment(4, N_min=10) is False, "FAIL: Sec 8.4 failed to trigger C4 fallback on collapsed inliers."
    # ---------------------------------------------------------
    # TEST 2: Section 8.4 Unconstrained Search Window
    # ---------------------------------------------------------
    N_o = 12
    fallback_window = calculate_fallback_search_window(N_o)
    # The fallback must strictly cover exactly 12 bins representing the full 360 degrees
    assert len(fallback_window) == 12, f"FAIL: Fallback window must sweep all {N_o} orientations."
    assert fallback_window[0] == 0, "FAIL: Fallback window must start at 0."
    assert fallback_window[-1] == 11, "FAIL: Fallback window must end at 11."

from patchrift.utils.complexity import calculate_section9_complexity

def test_section_9_computational_bounds():
    # Simulate processing a true Chandrayaan-2 OHRC strip (e.g., 12000x12000 pixels)
    # extracting K=500 keypoints of size 128x128.
    speed_multiplier, mem_full_mb, mem_patch_mb = calculate_section9_complexity(
        image_dim=12000, 
        P=128, 
        K=500
    )
    
    # Section 9 mandates a roughly thirtyfold reduction for OHRC strip specifications
    assert speed_multiplier > 30.0, f"FAIL: Architecture is too slow. Expected >30x gain, got {speed_multiplier}x"
    
    # Section 9 mandates strict memory limits. 
    # A full OHRC image Log-Gabor bank requires over 100,000 MB (100 GB) of RAM.
    assert mem_full_mb > 100000.0, "FAIL: Full image memory calculation incorrect."
    
    # Patch-based must require less than 15 MB of peak RAM.
    assert mem_patch_mb < 15.0, f"FAIL: Patch architecture consumes too much memory: {mem_patch_mb} MB"
