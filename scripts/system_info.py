import platform

import cv2
import numpy as np
import scipy
import spiceypy
import pyfftw


print("=== Patch-RIFT Environment ===")
print()

print(f"Python      : {platform.python_version()}")
print(f"Platform    : {platform.platform()}")
print(f"Machine     : {platform.machine()}")

print()
print("=== Scientific Libraries ===")

print(f"NumPy       : {np.__version__}")
print(f"SciPy       : {scipy.__version__}")
print(f"OpenCV      : {cv2.__version__}")
print(f"SpiceyPy    : {spiceypy.__version__}")
print(f"pyFFTW      : {pyfftw.__version__}")