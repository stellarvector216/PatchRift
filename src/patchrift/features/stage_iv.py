import math
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter
from dataclasses import dataclass
from patchrift.utils.complexity import use_full_image_formulation

def create_log_gabor_bank(rows, cols, num_scales=4, num_orientations=12, 
                          min_wavelength=3, mult=2.0, sigma_on_f=0.55):
    """
    Stage IV: Generates a Log-Gabor filter bank in the frequency domain.
    Strictly implements Section 7.1 (Equations 48-50) of the RIFT specification.
    """
    if any(int(value) != value or value <= 0 for value in (rows, cols, num_scales, num_orientations)):
        raise ValueError("filter dimensions, scales, and orientations must be positive integers")
    if not np.isfinite(min_wavelength) or min_wavelength <= 0.0 or not np.isfinite(mult) or mult <= 1.0:
        raise ValueError("min_wavelength must be positive and scale multiplier greater than one")
    if not np.isfinite(sigma_on_f) or not 0.0 < sigma_on_f < 1.0:
        raise ValueError("sigma_on_f must be between zero and one")
    # 1. Construct the polar coordinate frequency grid
    # Shift the zero-frequency component to the center of the spectrum
    y, x = np.mgrid[-rows//2 : rows//2, -cols//2 : cols//2]
    
    # Normalize coordinates to [-0.5, 0.5]
    y = y / rows
    x = x / cols
    
    # Calculate radius (frequency) and angle for every point
    radius = np.sqrt(x**2 + y**2)
    radius[rows//2, cols//2] = 1.0  # Temporarily set center to 1 to avoid log(0)
    theta = np.arctan2(y, x)
    
    # Shift zero-frequency back to the corners (standard FFT layout)
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    
    # Zero out the DC component exactly to prevent brightness leakage
    radius[0, 0] = 0.0
    
    # 2. Pre-calculate the angular components (Equation 49)
    # sigma_theta = pi / (1.2 * N_o)
    sigma_theta = np.pi / (1.2 * num_orientations)
    angular_filters = []
    
    for o in range(num_orientations):
        # psi_o = pi * (o - 1) / N_o  -> Zero-indexed in Python: pi * o / N_o
        angle_center = o * np.pi / num_orientations
        
        # Calculate angular difference, wrapped to [-pi/2, pi/2]
        delta_theta = theta - angle_center
        delta_theta = np.mod(delta_theta + np.pi/2, np.pi) - np.pi/2
        
        # Compute the angular Gaussian
        angular_component = np.exp(-(delta_theta**2) / (2 * sigma_theta**2))
        angular_filters.append(angular_component)
        
    # 3. Pre-calculate the radial components and combine (Equation 48)
    # bandwidth B_w = 2 octaves requires sigma_on_f = 0.55 (Equation 50)
    log_gabor_bank = []
    
    for s in range(num_scales):
        # Center wavelength for this scale: lambda_s = lambda_min * mult^(s)
        wavelength = min_wavelength * (mult ** s)
        center_freq = 1.0 / wavelength
        
        # Compute the radial Log-Gabor component
        # Warn: Ignore divide by zero for the DC component, we force it to 0 later
        with np.errstate(divide='ignore', invalid='ignore'):
            radial_component = np.exp(-(np.log(radius / center_freq)**2) / (2 * np.log(sigma_on_f)**2))
        
        # Enforce exactly zero DC response
        radial_component[0, 0] = 0.0
        
        # Multiply radial and angular components to create the 2D filter
        scale_filters = []
        for o in range(num_orientations):
            filter_2d = radial_component * angular_filters[o]
            scale_filters.append(filter_2d)
            
        log_gabor_bank.append(scale_filters)
        
    # log_gabor_bank[scale][orientation]
    return log_gabor_bank

def calculate_phase_congruency(image_patch, filter_bank, k=2.5, c_W=0.5, gamma_W=10, epsilon=1e-4, return_amplitudes=False):
    """
    Implements Equations 51, 52, 53, and 54 of the RIFT specification.
    Computes Phase Congruency per orientation from the Log-Gabor filter bank.
    """
    image_patch = np.asarray(image_patch, dtype=float)
    if image_patch.ndim != 2 or not np.all(np.isfinite(image_patch)):
        raise ValueError("image_patch must be a finite 2D array")
    if not filter_bank or not filter_bank[0]:
        raise ValueError("filter_bank must contain at least one scale and orientation")
    if epsilon <= 0.0 or not np.isfinite(epsilon) or not np.isfinite(k):
        raise ValueError("epsilon must be positive and noise multiplier finite")
    num_scales = len(filter_bank)
    num_orientations = len(filter_bank[0])
    if any(len(scale) != num_orientations for scale in filter_bank):
        raise ValueError("every scale must contain the same number of orientations")
    if any(np.shape(response) != image_patch.shape for scale in filter_bank for response in scale):
        raise ValueError("every frequency response must match image_patch dimensions")
    
    # Precompute the image FFT (Equation 51 preparation)
    img_fft = np.fft.fft2(image_patch)
    
    PC = [] # Will hold PC_o for each orientation
    amplitude_sums = []
    
    for o in range(num_orientations):
        # Accumulators for this orientation across all scales
        sum_e = np.zeros_like(image_patch, dtype=float)
        sum_o = np.zeros_like(image_patch, dtype=float)
        sum_A = np.zeros_like(image_patch, dtype=float)
        
        A_so_list = []
        phi_so_list = []
        
        # Equation 51 requires an analytic response. The stored Log-Gabor
        # magnitudes are pi-symmetric, so retain only the oriented half-plane
        # and double its amplitude to form the quadrature pair.
        angle = o * np.pi / num_orientations
        analytic = _analytic_half_plane(*image_patch.shape, angle)

        for s in range(num_scales):
            # EQUATION 51: Filtered analytic signal via inverse FFT
            filtered_fft = img_fft * filter_bank[s][o] * analytic
            signal = np.fft.ifft2(filtered_fft)
            
            e_so = signal.real
            o_so = signal.imag
            A_so = np.abs(signal)
            phi_so = np.arctan2(o_so, e_so)
            
            A_so_list.append(A_so)
            phi_so_list.append(phi_so)
            
            sum_e += e_so
            sum_o += o_so
            sum_A += A_so

        amplitude_sums.append(sum_A)
            
        # EQUATION 54: Robust noise floor estimation (T_o)
        # Derived from the Rayleigh distribution of the smallest-scale amplitude (s=0)
        A_smallest = A_so_list[0]
        tau_R = np.median(A_smallest) / np.sqrt(np.log(4))
        mu_R = tau_R * np.sqrt(np.pi / 2)
        sigma_R = tau_R * np.sqrt((4 - np.pi) / 2)
        T_o = mu_R + k * sigma_R
        
        # EQUATION 53: Frequency-spread weight (W_o)
        # Suppresses responses concentrated in a single band (e.g., sensor striping)
        max_A = np.max(A_so_list, axis=0)
        s_o = (1.0 / num_scales) * (sum_A / (max_A + epsilon))
        W_o = 1.0 / (1.0 + np.exp(gamma_W * (c_W - s_o)))
        
        # EQUATION 52: Phase Congruency Formulation (PC_o)
        # Amplitude-weighted mean phase across scales
        mean_phi_o = np.arctan2(sum_o, sum_e)
        
        PC_numerator = np.zeros_like(image_patch, dtype=float)
        
        for s in range(num_scales):
            # Phase deviation Delta Phi_so
            phase_diff = phi_so_list[s] - mean_phi_o
            delta_phi_so = np.cos(phase_diff) - np.abs(np.sin(phase_diff))
            
            # Apply zero-floor clamp to the noise-subtracted response (the |_ _|_{+} notation)
            term = A_so_list[s] * delta_phi_so - T_o
            term = np.maximum(term, 0.0)
            
            PC_numerator += W_o * term
            
        # Final Phase Congruency mapping for this specific orientation
        PC_o = PC_numerator / (sum_A + epsilon)
        PC.append(PC_o)
        
    if return_amplitudes:
        return PC, np.stack(amplitude_sums, axis=0)
    return PC


def _analytic_half_plane(rows, cols, angle):
    """Return the oriented analytic-spectrum factor for a real filter bank."""
    fy = np.fft.fftfreq(rows)[:, None]
    fx = np.fft.fftfreq(cols)[None, :]
    directional_frequency = fx * np.cos(angle) + fy * np.sin(angle)
    factor = np.where(directional_frequency > 0.0, 2.0, 0.0)
    factor[np.isclose(directional_frequency, 0.0)] = 1.0
    factor[0, 0] = 0.0
    return factor

def calculate_moment_analysis(PC_list):
    """
    Implements Section 7.2 (Equations 55 and 56).
    Extracts cornerness (M_min) and edgeness (M_max) from Phase Congruency.
    """
    num_orientations = len(PC_list)
    
    # Initialize the orientation-energy covariance terms (Equation 55)
    a = np.zeros_like(PC_list[0], dtype=float)
    b = np.zeros_like(PC_list[0], dtype=float)
    c = np.zeros_like(PC_list[0], dtype=float)
    
    for o in range(num_orientations):
        psi_o = o * np.pi / num_orientations
        
        pc_cos = PC_list[o] * np.cos(psi_o)
        pc_sin = PC_list[o] * np.sin(psi_o)
        
        a += pc_cos ** 2
        b += 2 * (pc_cos * pc_sin)
        c += pc_sin ** 2
        
    # Calculate eigenvalues for cornerness and edgeness (Equation 56)
    term1 = a + c
    term2 = np.sqrt(b**2 + (a - c)**2)
    
    M_max = 0.5 * (term1 + term2)
    M_min = 0.5 * (term1 - term2)
    
    return M_max, M_min

def principal_curvature_ratio_test(M_min, r=10):
    """
    Implements Section 7.3 (Equation 58).
    Evaluates the Hessian of M_min to reject keypoints along straight edges.
    """
    # Calculate gradients of M_min to build the Hessian matrix H
    dy, dx = np.gradient(M_min)
    dyy, dyx = np.gradient(dy)
    dxy, dxx = np.gradient(dx)
    
    # Trace and Determinant of the Hessian
    Tr_H = dxx + dyy
    Det_H = (dxx * dyy) - (dxy * dyx)
    
    # Prevent division by zero in flat regions
    Det_H[Det_H == 0] = 1e-8
    
    # Equation 58 constraint using the curvature ratio threshold 'r'
    threshold = ((r + 1.0)**2) / r
    
    # A valid corner must be a local extremum (Det_H > 0) 
    # and not highly elongated (ratio < threshold)
    valid_curvature_mask = (Det_H > 0) & ((Tr_H**2 / Det_H) < threshold)
    
    return valid_curvature_mask

def validate_downsample_factor(D_min, lambda_min_prime, d_proposed):
    """
    Implements Section 7.2 (Equation 57).
    Validates that the detector downsample factor 'd' preserves feature survival.
    """
    # Equation 57: The crater must remain detectable after decimation
    d_max = D_min / (4.0 * lambda_min_prime)
    
    if d_proposed > d_max:
        raise ValueError(f"Downsample factor d={d_proposed} exceeds the survival bound of {d_max:.2f}. "
                         f"Features of size D_min={D_min} will be destroyed.")
    
    return True

def next_5_smooth(n):
    """
    Finds the smallest 5-smooth integer (prime factors of only 2, 3, 5) >= n.
    This strictly ensures the Fast Fourier Transform operates at peak efficiency.
    """
    def is_5_smooth(x):
        if x <= 0: return False
        for p in [2, 3, 5]:
            while x % p == 0:
                x //= p
        return x == 1
    
    n = int(math.ceil(n))
    while not is_5_smooth(n):
        n += 1
    return n

def calculate_patch_size_bounds(lambda_max, r_max, m=4, B_omega=2.0, r_taper=0.25):
    """
    Implements Section 7.4 (Equations 59 and 60).
    Calculates the minimum required patch size P to satisfy both spectral 
    and spatial constraints, returning a 5-smooth integer.
    """
    # EQUATION 59: Spectral resolution bound
    # Ensures the longest-wavelength channel is represented by at least 'm' frequency bins
    denominator = (2**(B_omega / 2.0)) - (2**(-B_omega / 2.0))
    P_spectral = (m * lambda_max) / denominator
    
    # EQUATION 60: Spatial support bound
    # Ensures the outermost pooling radius (r_max) lies strictly inside the untapered region
    P_spatial = (2.0 * r_max) / (1.0 - r_taper)
    
    # The final patch size must satisfy both bounds
    P_min = max(P_spectral, P_spatial)
    
    # Round up to the nearest 5-smooth integer as mandated by the architecture
    P_final = next_5_smooth(P_min)
    
    return P_final, P_spectral, P_spatial

def compute_sampling_matrix(sigma_X, alpha_X):
    """
    Implements Section 7.5 (Equation 61).
    Constructs the sampling matrix M_X that maps the canonical frame 
    to the native image frame, removing rotation and scale analytically.
    """
    # Equation 61: M_X = sigma_X * R(alpha_X + pi/2)
    theta = alpha_X + (np.pi / 2.0)
    
    # R(theta) is the standard 2D rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    M_X = sigma_X * R
    return M_X

def generate_tukey_window(P, r_taper):
    """
    Implements Section 7.5 (Equation 63).
    Constructs the 2D Tukey window applied to the canonical patch.
    """
    v = np.arange(-P // 2, P // 2)
    w_1d = np.zeros(P)
    
    # Untapered region boundary
    threshold = (P / 2.0) * (1.0 - r_taper)
    
    for idx, v_i in enumerate(v):
        abs_v = np.abs(v_i)
        if abs_v <= threshold:
            w_1d[idx] = 1.0
        else:
            # Equation 63 cosine taper term
            term = (np.pi / r_taper) * ((abs_v / (P / 2.0)) - 1.0 + r_taper)
            w_1d[idx] = 0.5 * (1.0 + np.cos(term))
            
    # Equation 63 dictates the product over i in {x, y}
    # This is mathematically the outer product of the 1D window with itself
    w_2d = np.outer(w_1d, w_1d)
    return w_2d

def extract_canonical_patch(I_cond, x_k, y_k, M_X, P, r_taper=0.25):
    """
    Implements Section 7.5 (Equation 62).
    Extracts the canonical patch using a single resampling pass.
    """
    # 1. Define the canonical grid v in Lambda = {-P/2, ..., P/2 - 1}^2
    v_range = np.arange(-P // 2, P // 2)
    V_X, V_Y = np.meshgrid(v_range, v_range)
    
    # Flatten the grid coordinates for matrix multiplication
    V_flat = np.vstack((V_X.flatten(), V_Y.flatten()))
    
    # 2. Map canonical coordinates to native image coordinates: x_k + M_X * v
    native_coords = M_X @ V_flat
    
    target_x = x_k + native_coords[0, :]
    target_y = y_k + native_coords[1, :]
    
    # map_coordinates expects (row, col) indexing, which maps to (y, x)
    coords = np.vstack((target_y, target_x))
    
    # 3. The specification requires Lanczos-3, not cubic-spline interpolation.
    patch_raw = lanczos3_interpolate(I_cond, target_x.reshape((P, P)), target_y.reshape((P, P)))
    
    # 4. Apply the Tukey window w(v)
    w = generate_tukey_window(P, r_taper)
    I_patch = w * patch_raw
    
    return I_patch


def lanczos3_interpolate(image, x, y):
    """Sample with a separable, affine-reproducing Lanczos-3 kernel.

    The six-tap Lanczos-3 weights are locally corrected to have unit sum and
    zero first moment. This preserves the windowed-sinc interpolation while
    making constant and affine image fields exact. At the array boundary,
    unavailable taps are omitted and the remaining weights are renormalized;
    coordinates outside the image are returned as zero and should generally
    be rejected by the caller's validity mask.
    """
    image = np.asarray(image, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if image.ndim != 2 or x.shape != y.shape:
        raise ValueError("image must be 2D and x/y coordinates must have matching shapes")
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("image and interpolation coordinates must be finite")

    ix, wx = _lanczos3_weights(x, image.shape[1])
    iy, wy = _lanczos3_weights(y, image.shape[0])
    result = np.zeros(x.shape, dtype=float)
    inside_coordinate = (
        (x >= 0.0) & (x <= image.shape[1] - 1)
        & (y >= 0.0) & (y <= image.shape[0] - 1)
    )
    for ky in range(6):
        for kx in range(6):
            weight = wy[..., ky] * wx[..., kx]
            active = inside_coordinate & (weight != 0.0)
            if np.any(active):
                result[active] += weight[active] * image[iy[..., ky][active], ix[..., kx][active]]
    return result


def _lanczos3_weights(coordinates, length):
    """Return bounded six-tap weights with exact zeroth/first moments."""
    base = np.floor(coordinates).astype(np.int64)
    indexes = base[..., None] + np.arange(-2, 4, dtype=np.int64)
    offsets = coordinates[..., None] - indexes
    active = (indexes >= 0) & (indexes < length) & (np.abs(offsets) < 3.0)
    weights = np.sinc(offsets) * np.sinc(offsets / 3.0)
    weights = np.where(active, weights, 0.0)

    total = np.sum(weights, axis=-1, keepdims=True)
    weights = np.divide(weights, total, out=np.zeros_like(weights), where=np.abs(total) > 1e-12)

    count = np.sum(active, axis=-1, keepdims=True)
    mean_offset = np.divide(
        np.sum(np.where(active, offsets, 0.0), axis=-1, keepdims=True),
        count,
        out=np.zeros_like(total),
        where=count > 0,
    )
    direction = np.where(active, offsets - mean_offset, 0.0)
    correction_denominator = np.sum(direction * offsets, axis=-1, keepdims=True)
    first_moment = np.sum(weights * offsets, axis=-1, keepdims=True)
    correction = np.divide(
        -first_moment,
        correction_denominator,
        out=np.zeros_like(first_moment),
        where=np.abs(correction_denominator) > 1e-12,
    )
    weights = np.where(active, weights + correction * direction, 0.0)
    return indexes, weights

def extract_canonical_mask(mask, x_k, y_k, M_X, P):
    """
    Extracts masks through the identical M_X using nearest-neighbour sampling,
    as explicitly mandated by Section 7.5.
    """
    v_range = np.arange(-P // 2, P // 2)
    V_X, V_Y = np.meshgrid(v_range, v_range)
    V_flat = np.vstack((V_X.flatten(), V_Y.flatten()))
    
    native_coords = M_X @ V_flat
    target_x = x_k + native_coords[0, :]
    target_y = y_k + native_coords[1, :]
    
    coords = np.vstack((target_y, target_x))
    
    # order=0 strictly enforces nearest-neighbor interpolation to preserve boolean mask integrity
    patch_mask = map_coordinates(mask, coords, order=0, mode='constant', cval=0.0)
    return patch_mask.reshape((P, P)).astype(bool)


def quadratic_subpixel_peak(response, center_x, center_y, window_size):
    """Locate the strongest local response and fit its negative quadratic peak.

    Returns ``(integer_dx, integer_dy, fractional_dx, fractional_dy)`` relative
    to the requested centre, or ``None`` when a safe 3x3 quadratic fit is not
    available. Invalid/non-finite samples are never allowed into the solve.
    """
    response = np.asarray(response, dtype=float)
    if response.ndim != 2 or not np.isfinite(center_x) or not np.isfinite(center_y):
        return None
    if not np.isfinite(window_size) or window_size < 1:
        return None
    if not np.all(np.isfinite(response)):
        return None
    radius = int(np.floor(window_size / 2.0))
    cx, cy = int(round(center_x)), int(round(center_y))
    x0, x1 = max(0, cx - radius), min(response.shape[1], cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(response.shape[0], cy + radius + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    local = response[y0:y1, x0:x1]
    py0, px0 = np.unravel_index(int(np.argmax(local)), local.shape)
    px, py = x0 + px0, y0 + py0
    if px <= 0 or px >= response.shape[1] - 1 or py <= 0 or py >= response.shape[0] - 1:
        return None
    z = response
    f0 = z[py, px]
    gx = 0.5 * (z[py, px + 1] - z[py, px - 1])
    gy = 0.5 * (z[py + 1, px] - z[py - 1, px])
    hxx = z[py, px + 1] - 2.0 * f0 + z[py, px - 1]
    hyy = z[py + 1, px] - 2.0 * f0 + z[py - 1, px]
    hxy = 0.25 * (z[py + 1, px + 1] - z[py + 1, px - 1] - z[py - 1, px + 1] + z[py - 1, px - 1])
    hessian = np.array([[hxx, hxy], [hxy, hyy]], dtype=float)
    gradient = np.array([gx, gy], dtype=float)
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(gradient)):
        return None
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.any(eigenvalues >= -1e-12):
        return None
    try:
        delta = -np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(delta)):
        return None
    return float(px - center_x), float(py - center_y), float(delta[0]), float(delta[1])

def generate_log_polar_layout(P=128, r1=19, r2=35, rmax=48):
    """
    Implements Section 7.6 log-polar spatial layout.
    Generates a J=17 cell layout: 1 central disc and 2 annuli of 8 sectors each.
    """
    v_range = np.arange(-P // 2, P // 2)
    X, Y = np.meshgrid(v_range, v_range)
    
    radius = np.sqrt(X**2 + Y**2)
    # atan2(Y,X) mapped strictly to [0, 2pi)
    angle = np.mod(np.arctan2(Y, X), 2 * np.pi)
    
    cell_map = np.full((P, P), -1, dtype=int)
    
    # Sector indices (0 to 7) based on 45-degree (pi/4) boundaries
    sector = np.floor(angle / (np.pi / 4.0)).astype(int)
    sector = np.clip(sector, 0, 7)
    
    # Cell 0: Central disc
    cell_map[radius <= r1] = 0
    
    # Cells 1-8: First annulus
    mask_annulus1 = (radius > r1) & (radius <= r2)
    cell_map[mask_annulus1] = 1 + sector[mask_annulus1]
    
    # Cells 9-16: Second annulus
    mask_annulus2 = (radius > r2) & (radius <= rmax)
    cell_map[mask_annulus2] = 9 + sector[mask_annulus2]
    
    return cell_map

def compute_maximum_index_map(PC_list):
    """
    Implements Section 7.6 (Equation 64).
    M(v) = argmax_o PC_o(v)
    Returns the integer map of the dominant structural orientation.
    """
    PC_stack = np.stack(PC_list, axis=0)
    MIM = np.argmax(PC_stack, axis=0)
    return MIM

def extract_descriptor_and_weights(MIM, A_sum_stack, cell_map, R_mask, S_int_mask, N_o=12, v_min=0.2, clamp_val=0.2):
    """
    Implements Section 7.6 (Equations 65 and 66).
    Builds the 204-dimensional descriptor (17 cells * 12 orientations) and 
    calculates the 17-dimensional reliability weight vector.
    
    A_sum_stack: Array of shape (N_o, P, P) containing sum of amplitudes across scales.
    """
    MIM = np.asarray(MIM)
    A_sum_stack = np.asarray(A_sum_stack, dtype=float)
    cell_map = np.asarray(cell_map)
    R_mask, S_int_mask = np.asarray(R_mask, dtype=bool), np.asarray(S_int_mask, dtype=bool)
    if MIM.ndim != 2 or A_sum_stack.shape != (N_o, *MIM.shape):
        raise ValueError("MIM and amplitude stack dimensions are inconsistent")
    if any(mask.shape != MIM.shape for mask in (cell_map, R_mask, S_int_mask)):
        raise ValueError("cell map and masks must match MIM dimensions")
    if not np.all(np.isfinite(A_sum_stack)) or np.any(A_sum_stack < 0.0):
        raise ValueError("amplitudes must be finite and nonnegative")
    if N_o <= 0 or not 0.0 <= v_min <= 1.0 or not 0.0 < clamp_val <= 1.0:
        raise ValueError("N_o, v_min, or clamp_val is outside its valid range")
    J = 17
    descriptor = np.zeros(J * N_o)
    reliability_weights = np.zeros(J)
    
    # Extract the amplitude sum specifically for the winning orientation at each pixel
    rows, cols = MIM.shape
    row_indices, col_indices = np.indices((rows, cols))
    winning_A_sum = A_sum_stack[MIM, row_indices, col_indices]
    
    for j in range(J):
        # Mask isolating current geometric cell
        C_j_mask = (cell_map == j)
        C_j_area = np.sum(C_j_mask)
        
        if C_j_area == 0:
            continue
            
        # --- EQUATION 65 (Descriptor Construction) ---
        MIM_j = MIM[C_j_mask]
        A_j = winning_A_sum[C_j_mask]
        
        # Histogram weighted by the amplitude of the dominant phase
        hist, _ = np.histogram(MIM_j, bins=N_o, range=(0, N_o), weights=A_j)
        
        # Per-cell L2 normalization, clamping, and renormalization
        norm = np.linalg.norm(hist)
        if norm > 1e-8:
            hist = hist / norm
            hist = np.clip(hist, 0.0, clamp_val)
            hist = hist / np.linalg.norm(hist)
            
        descriptor[j * N_o : (j + 1) * N_o] = hist
        
        # --- EQUATION 66 (Reliability Weights) ---
        # Penalize cells containing shadow interiors or cast-shadow terminators
        R_overlap = np.sum(R_mask[C_j_mask])
        S_int_overlap = np.sum(S_int_mask[C_j_mask])
        
        term1 = max(0.0, 1.0 - (R_overlap / C_j_area))
        term2 = max(0.0, 1.0 - (S_int_overlap / C_j_area))
        
        # Floor bound ensures we preserve a minimum contribution from self-similar terrain
        reliability_weights[j] = max(term1 * term2, v_min)
        
    return descriptor, reliability_weights


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    response: float


@dataclass(frozen=True)
class FeatureSet:
    keypoints: tuple[Keypoint, ...]
    canonical_positions: np.ndarray
    descriptors: np.ndarray
    reliability: np.ndarray
    scale_ratio: float
    sampling_matrix: np.ndarray


def detect_keypoints(
    image,
    valid_mask,
    shadow_interior,
    *,
    downsample=8,
    min_wavelength=3.0,
    min_feature_diameter=96.0,
    coarse_response_percentile=50.0,
    grid_size=64,
    per_cell=2,
    target_count=500,
    curvature_ratio=10.0,
    patch_size=128,
    shadow_fraction_max=0.40,
):
    """Run the §7.1-7.3 detector and return working-grid (x,y) keypoints."""
    image = np.asarray(image, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    interior = np.asarray(shadow_interior, dtype=bool)
    if image.ndim != 2 or valid.shape != image.shape or interior.shape != image.shape:
        raise ValueError("image and both masks must be matching 2D arrays")
    if downsample <= 0 or grid_size <= 0 or per_cell <= 0 or target_count <= 0:
        raise ValueError("downsample and keypoint limits must be positive")
    if not 0.0 <= coarse_response_percentile <= 100.0 or not 0.0 <= shadow_fraction_max <= 1.0:
        raise ValueError("percentile and shadow fraction must lie within [0, 100] and [0, 1]")
    validate_downsample_factor(min_feature_diameter, min_wavelength, downsample)
    detector = _sample_decimated_grid(image, downsample, order=1, mode="nearest")
    detector_valid = _sample_decimated_grid(valid.astype(np.uint8), downsample, order=0, mode="constant").astype(bool)
    bank = create_log_gabor_bank(
        *detector.shape, num_scales=3, num_orientations=4,
        min_wavelength=min_wavelength,
    )
    pc = calculate_phase_congruency(detector, bank)
    _, cornerness = calculate_moment_analysis(pc)
    candidates = _harris_candidates(cornerness)
    values = cornerness[detector_valid & np.isfinite(cornerness)]
    if values.size == 0:
        return ()
    cutoff = float(np.percentile(values, coarse_response_percentile))
    eligible = candidates & detector_valid & (cornerness >= cutoff)
    eligible &= principal_curvature_ratio_test(cornerness, r=curvature_ratio)
    candidate_rows, candidate_cols = np.nonzero(eligible)
    if candidate_rows.size == 0:
        return ()

    # Map the shadow-interior mask onto detector coordinates, then reject a
    # proposal if too much of its eventual canonical patch falls inside S_int.
    detector_interior = _sample_decimated_grid(
        interior.astype(np.uint8), downsample, order=0, mode="nearest"
    ).astype(bool)
    half = patch_size // 2
    candidates_out = []
    for row, col in zip(candidate_rows, candidate_cols):
        y_work = float(row * downsample)
        x_work = float(col * downsample)
        r0 = max(0, int(row) - half // downsample)
        r1 = min(detector_interior.shape[0], int(row) + half // downsample + 1)
        c0 = max(0, int(col) - half // downsample)
        c1 = min(detector_interior.shape[1], int(col) + half // downsample + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        if np.mean(detector_interior[r0:r1, c0:c1]) > shadow_fraction_max:
            continue
        candidates_out.append(Keypoint(x_work, y_work, float(cornerness[row, col])))

    # At most ncell proposals are retained in each 64x64 working-grid cell.
    by_cell: dict[tuple[int, int], list[Keypoint]] = {}
    for point in candidates_out:
        key = (int(point.y // grid_size), int(point.x // grid_size))
        by_cell.setdefault(key, []).append(point)
    distributed = []
    for points in by_cell.values():
        distributed.extend(sorted(points, key=lambda point: point.response, reverse=True)[:per_cell])
    return tuple(sorted(distributed, key=lambda point: point.response, reverse=True)[:target_count])


def _sample_decimated_grid(array, factor, *, order, mode):
    """Decimate at exact ``factor`` source-pixel intervals, not zoom's rounded ratio."""
    array = np.asarray(array)
    out_rows = int(np.floor((array.shape[0] - 1) / factor)) + 1
    out_cols = int(np.floor((array.shape[1] - 1) / factor)) + 1
    rows, cols = np.mgrid[:out_rows, :out_cols]
    return map_coordinates(
        array, np.stack((rows * factor, cols * factor)), order=order,
        mode=mode, cval=0.0, prefilter=order > 1,
    )


def _harris_candidates(cornerness):
    gy, gx = np.gradient(np.asarray(cornerness, dtype=float))
    xx = gaussian_filter(gx * gx, 1.0)
    yy = gaussian_filter(gy * gy, 1.0)
    xy = gaussian_filter(gx * gy, 1.0)
    response = (xx * yy - xy * xy) - 0.04 * (xx + yy) ** 2
    return (response == maximum_filter(response, size=3, mode="nearest")) & (response > 0.0)


def describe_keypoints(
    keypoints,
    conditioned_native,
    native_masks,
    alpha,
    scale_ratio,
    *,
    patch_size=128,
    lambda_min=3.0,
    scale_multiplier=2.0,
    num_scales=4,
    num_orientations=12,
    r1=19.0,
    r2=35.0,
    rmax=48.0,
    taper=0.25,
    v_min=0.2,
    downsample=8,
    native_coordinate_scale: tuple[float, float] | None = None,
):
    """Build canonical PC/MIM descriptors and Eq. (66) mask reliability."""
    if not np.isfinite(scale_ratio) or scale_ratio <= 0.0:
        raise ValueError("scale_ratio must be finite and positive")
    if downsample <= 0:
        raise ValueError("downsample must be positive")
    if native_coordinate_scale is None:
        native_coordinate_scale = (float(scale_ratio), float(scale_ratio))
    if len(native_coordinate_scale) != 2 or not np.all(np.isfinite(native_coordinate_scale)) or min(native_coordinate_scale) <= 0.0:
        raise ValueError("native_coordinate_scale must contain two finite positive values")
    x_scale, y_scale = map(float, native_coordinate_scale)
    matrix = compute_sampling_matrix(scale_ratio, alpha)
    cell_map = generate_log_polar_layout(patch_size, r1, r2, rmax)
    if use_full_image_formulation(
        np.asarray(conditioned_native).size, patch_size, len(keypoints)
    ):
        return _describe_keypoints_full_image(
            keypoints,
            conditioned_native,
            native_masks,
            matrix,
            scale_ratio,
            x_scale,
            y_scale,
            patch_size=patch_size,
            lambda_min=lambda_min,
            scale_multiplier=scale_multiplier,
            num_scales=num_scales,
            num_orientations=num_orientations,
            cell_map=cell_map,
            r1=r1,
            r2=r2,
            rmax=rmax,
            taper=taper,
            v_min=v_min,
            downsample=downsample,
        )
    bank = create_log_gabor_bank(
        patch_size, patch_size, num_scales, num_orientations,
        min_wavelength=lambda_min, mult=scale_multiplier,
    )
    descriptor_rows = []
    reliability_rows = []
    canonical_positions = []
    kept_points = []
    valid_native = np.asarray(native_masks.get("valid", np.ones_like(conditioned_native, dtype=bool)), dtype=bool)
    descriptor_downsample = int(downsample)
    for point in keypoints:
        seed_x, seed_y = point.x * x_scale, point.y * y_scale
        original_seed_x, original_seed_y = seed_x, seed_y

        # Eq. (62) samples the whole canonical patch. The source defines no
        # no-data reliability term, so require the patch and Lanczos support
        # to be observed instead of silently treating missing data as terrain.
        if not _canonical_descriptor_support_is_valid(
            valid_native, seed_x, seed_y, matrix, patch_size / 2.0
        ):
            continue

        def response_at(x_native, y_native):
            patch_value = extract_canonical_patch(
                conditioned_native, x_native, y_native, matrix, patch_size, taper
            )
            pc_value, amplitudes = calculate_phase_congruency(patch_value, bank, return_amplitudes=True)
            _, minimum = calculate_moment_analysis(pc_value)
            return patch_value, pc_value, amplitudes, minimum

        patch, pc, amplitude_sum, minimum = response_at(seed_x, seed_y)
        centre = patch_size // 2
        refined = quadratic_subpixel_peak(minimum, centre, centre, descriptor_downsample)
        if refined is None:
            continue
        coarse_dx, coarse_dy, sub_dx, sub_dy = refined
        displacement = np.array([coarse_dx + sub_dx, coarse_dy + sub_dy])
        if np.max(np.abs(displacement)) > 0.5:
            # One recenter/recompute is the specified correction for a
            # displacement larger than half a canonical pixel.
            recenter = matrix @ displacement
            seed_x += float(recenter[0])
            seed_y += float(recenter[1])
            patch, pc, amplitude_sum, minimum = response_at(seed_x, seed_y)
            retry = quadratic_subpixel_peak(minimum, centre, centre, descriptor_downsample)
            if retry is None:
                continue
            rdx, rdy, rsubx, rsuby = retry
            displacement += np.array([rdx + rsubx, rdy + rsuby])
            if max(abs(rsubx), abs(rsuby)) > 0.5:
                continue
        if np.max(np.abs(displacement)) > descriptor_downsample / 2.0:
            continue
        native_shift = matrix @ displacement
        x_native = original_seed_x + float(native_shift[0])
        y_native = original_seed_y + float(native_shift[1])
        if not _canonical_descriptor_support_is_valid(
            valid_native, x_native, y_native, matrix, patch_size / 2.0
        ):
            continue

        # For a small (<0.5 px) update, the existing patch is retained; when
        # recentered above, ``patch`` is already centered on the corrected seed.
        mim = compute_maximum_index_map(pc)
        r_patch = extract_canonical_mask(
            native_masks["R"], x_native, y_native, matrix, patch_size
        )
        interior_patch = extract_canonical_mask(
            native_masks["S_int"], x_native, y_native, matrix, patch_size
        )
        descriptor, weights = extract_descriptor_and_weights(
            mim, amplitude_sum, cell_map, r_patch, interior_patch,
            N_o=num_orientations, v_min=v_min,
        )
        q = np.linalg.solve(matrix, np.array([x_native, y_native]))
        canonical_positions.append(q)
        descriptor_rows.append(descriptor)
        reliability_rows.append(weights)
        kept_points.append(Keypoint(x_native / x_scale, y_native / y_scale, point.response))
    return FeatureSet(
        keypoints=tuple(kept_points),
        canonical_positions=np.asarray(canonical_positions, dtype=float).reshape((-1, 2)),
        descriptors=np.asarray(descriptor_rows, dtype=float).reshape((-1, 17 * num_orientations)),
        reliability=np.asarray(reliability_rows, dtype=float).reshape((-1, 17)),
        scale_ratio=float(scale_ratio),
        sampling_matrix=matrix,
    )


def _describe_keypoints_full_image(
    keypoints,
    conditioned_native,
    native_masks,
    matrix,
    scale_ratio,
    x_scale,
    y_scale,
    *,
    patch_size,
    lambda_min,
    scale_multiplier,
    num_scales,
    num_orientations,
    cell_map,
    r1,
    r2,
    rmax,
    taper,
    v_min,
    downsample,
):
    """Use one canonical full-image phase-congruency field for dense keys."""
    native = np.asarray(conditioned_native, dtype=float)
    valid_native = np.asarray(native_masks.get("valid", np.ones_like(native, bool)), dtype=bool)
    if native.ndim != 2 or valid_native.shape != native.shape:
        raise ValueError("conditioned image and native valid mask must be matching 2D arrays")
    matrix = np.asarray(matrix, dtype=float)
    native_center = np.array([(native.shape[1] - 1) / 2.0, (native.shape[0] - 1) / 2.0])
    corners = np.array([
        [0.0, 0.0], [native.shape[1] - 1.0, 0.0],
        [native.shape[1] - 1.0, native.shape[0] - 1.0],
        [0.0, native.shape[0] - 1.0],
    ])
    canonical_corners = np.linalg.solve(matrix, (corners - native_center).T).T
    q_min = np.floor(canonical_corners.min(axis=0)).astype(int)
    q_max = np.ceil(canonical_corners.max(axis=0)).astype(int)
    out_width, out_height = (q_max - q_min + 1).astype(int)
    rows, cols = np.mgrid[:out_height, :out_width]
    qx, qy = cols + q_min[0], rows + q_min[1]
    source = native_center[:, None] + matrix @ np.vstack((qx.ravel(), qy.ravel()))
    source_x = source[0].reshape(out_height, out_width)
    source_y = source[1].reshape(out_height, out_width)
    canonical_image = lanczos3_interpolate(native, source_x, source_y)
    canonical_image[~((source_x >= 0) & (source_x <= native.shape[1] - 1)
                      & (source_y >= 0) & (source_y <= native.shape[0] - 1))] = 0.0

    eligible = []
    for point in keypoints:
        x_native, y_native = point.x * x_scale, point.y * y_scale
        if _canonical_descriptor_support_is_valid(
            valid_native, x_native, y_native, matrix, patch_size / 2.0
        ):
            q = np.linalg.solve(matrix, np.array([x_native, y_native]) - native_center)
            eligible.append((point, x_native, y_native, float(q[0] - q_min[0]), float(q[1] - q_min[1])))

    if not eligible:
        return _empty_feature_set(scale_ratio, matrix, num_orientations)

    # The V3 density switch deliberately spends one image-wide bank/transform
    # to avoid recomputing the same PC field for heavily overlapping patches.
    bank = create_log_gabor_bank(
        out_height, out_width, num_scales, num_orientations,
        min_wavelength=lambda_min, mult=scale_multiplier,
    )
    pc, amplitude_sum = calculate_phase_congruency(
        canonical_image, bank, return_amplitudes=True
    )
    _, minimum = calculate_moment_analysis(pc)
    mim_global = compute_maximum_index_map(pc)
    v_range = np.arange(-patch_size // 2, patch_size // 2)
    vx, vy = np.meshgrid(v_range, v_range)
    descriptor_rows, reliability_rows, canonical_positions, kept_points = [], [], [], []

    for point, seed_native_x, seed_native_y, center_x, center_y in eligible:
        seed_center_x, seed_center_y = center_x, center_y
        refinement = quadratic_subpixel_peak(minimum, center_x, center_y, downsample)
        if refinement is None:
            continue
        dx, dy, fx, fy = refinement
        displacement = np.array([dx + fx, dy + fy])
        if np.max(np.abs(displacement)) > 0.5:
            # Re-evaluate around the updated seed. The retry displacement is
            # relative to this recentered location, so do not add the first
            # displacement a second time when computing the final location.
            center_x = seed_center_x + displacement[0]
            center_y = seed_center_y + displacement[1]
            retry = quadratic_subpixel_peak(minimum, center_x, center_y, downsample)
            if retry is None:
                continue
            rdx, rdy, rfx, rfy = retry
            displacement += np.array([rdx + rfx, rdy + rfy])
            if max(abs(rfx), abs(rfy)) > 0.5:
                continue
            refined_x, refined_y = center_x + rdx + rfx, center_y + rdy + rfy
        else:
            refined_x, refined_y = seed_center_x + displacement[0], seed_center_y + displacement[1]
        if np.max(np.abs(displacement)) > downsample / 2.0:
            continue

        q_world = np.array([refined_x + q_min[0], refined_y + q_min[1]])
        native_xy = native_center + matrix @ q_world
        x_native, y_native = map(float, native_xy)
        if not _canonical_descriptor_support_is_valid(
            valid_native, x_native, y_native, matrix, patch_size / 2.0
        ):
            continue

        sample_x = refined_x + vx
        sample_y = refined_y + vy
        coordinates = np.stack((sample_y, sample_x))
        mim = map_coordinates(mim_global, coordinates, order=0, mode="constant", cval=0.0).astype(int)
        amplitudes = np.stack([
            map_coordinates(amplitude_sum[o], coordinates, order=1, mode="constant", cval=0.0)
            for o in range(num_orientations)
        ])
        r_patch = extract_canonical_mask(
            native_masks["R"], x_native, y_native, matrix, patch_size
        )
        interior_patch = extract_canonical_mask(
            native_masks["S_int"], x_native, y_native, matrix, patch_size
        )
        descriptor, weights = extract_descriptor_and_weights(
            mim, amplitudes, cell_map, r_patch, interior_patch,
            N_o=num_orientations, v_min=v_min,
        )
        descriptor_rows.append(descriptor)
        reliability_rows.append(weights)
        # Match the patch formulation's canonical coordinate origin: native
        # image coordinates are mapped directly by M^{-1} in both branches.
        canonical_positions.append(np.linalg.solve(matrix, np.array([x_native, y_native])))
        kept_points.append(Keypoint(x_native / x_scale, y_native / y_scale, point.response))

    return FeatureSet(
        keypoints=tuple(kept_points),
        canonical_positions=np.asarray(canonical_positions, dtype=float).reshape((-1, 2)),
        descriptors=np.asarray(descriptor_rows, dtype=float).reshape((-1, 17 * num_orientations)),
        reliability=np.asarray(reliability_rows, dtype=float).reshape((-1, 17)),
        scale_ratio=float(scale_ratio),
        sampling_matrix=matrix,
    )


def _empty_feature_set(scale_ratio, matrix, num_orientations):
    return FeatureSet(
        keypoints=(),
        canonical_positions=np.empty((0, 2), dtype=float),
        descriptors=np.empty((0, 17 * num_orientations), dtype=float),
        reliability=np.empty((0, 17), dtype=float),
        scale_ratio=float(scale_ratio),
        sampling_matrix=matrix,
    )


def _canonical_descriptor_support_is_valid(valid_mask, x, y, matrix, radius, tap_radius=3):
    """Return whether descriptor support plus interpolation taps is observed."""
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 2 or not np.isfinite(x) or not np.isfinite(y):
        return False
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        return False
    corners = np.array(
        [[-radius, -radius], [radius, -radius], [radius, radius], [-radius, radius]],
        dtype=float,
    )
    source = (np.asarray(matrix, dtype=float) @ corners.T).T
    x0 = int(np.floor(x + source[:, 0].min())) - tap_radius
    x1 = int(np.ceil(x + source[:, 0].max())) + tap_radius + 1
    y0 = int(np.floor(y + source[:, 1].min())) - tap_radius
    y1 = int(np.ceil(y + source[:, 1].max())) + tap_radius + 1
    if x0 < 0 or y0 < 0 or x1 > valid.shape[1] or y1 > valid.shape[0]:
        return False
    return bool(np.all(valid[y0:y1, x0:x1]))
