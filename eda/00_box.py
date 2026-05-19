"""
Phase 0 EDA — Step 0.
Resolve LAMMPS triclinic bounds to true lattice vectors.

LAMMPS dump 'BOX BOUNDS xy xz yz' format:
  xlo_bound = xlo + min(0, xy, xz, xy+xz)
  xhi_bound = xhi + max(0, xy, xz, xy+xz)
  ylo_bound = ylo + min(0, yz)
  yhi_bound = yhi + max(0, yz)
True lattice vectors:
  A = (lx, 0, 0),  B = (xy, ly, 0),  C = (xz, yz, lz)
where lx = xhi - xlo, ly = yhi - ylo, lz = zhi - zlo.
"""
import numpy as np

# Raw values from dump header (final frame)
xlo_b, xhi_b, xy = 0.0, 1.9850840849634051e+02, 6.6169469498780174e+01
ylo_b, yhi_b, xz = 0.0, 1.1460888308176639e+02, 0.0
zlo_b, zhi_b, yz = 0.0, 3.5000000000000000e+01, 0.0

# Correct triclinic
xlo = xlo_b - min(0.0, xy, xz, xy + xz)
xhi = xhi_b - max(0.0, xy, xz, xy + xz)
ylo = ylo_b - min(0.0, yz)
yhi = yhi_b - max(0.0, yz)
lx, ly, lz = xhi - xlo, yhi - ylo, zhi_b - zlo_b

A = np.array([lx, 0.0])
B = np.array([xy, ly])
area = abs(np.cross(A, B))

print(f"xlo={xlo}, xhi={xhi}, lx={lx}")
print(f"ylo={ylo}, yhi={yhi}, ly={ly}")
print(f"A = {A}, |A| = {np.linalg.norm(A):.4f}")
print(f"B = {B}, |B| = {np.linalg.norm(B):.4f}")
print(f"angle(A,B) = {np.degrees(np.arccos(A@B / (np.linalg.norm(A)*np.linalg.norm(B)))):.3f} deg")
print(f"area = {area:.2f} A^2")

# Primitive cells per layer
a_hbn = 2.504
prim_area = (np.sqrt(3) / 2) * a_hbn ** 2
n_prim = area / prim_area
print(f"primitive cell area = {prim_area:.4f} A^2 (a={a_hbn})")
print(f"# primitive cells per layer (predicted from area) = {n_prim:.2f}")
print(f"# primitive cells per layer (atoms/2): 5582/2 = 2791")

# Moire period from |A| = L_moire = a / (2 sin(theta/2))
L = np.linalg.norm(A)
theta = 2 * np.arcsin(a_hbn / (2 * L))
print(f"L_moire = |A| = {L:.4f} A")
print(f"theta = {theta:.6f} rad = {np.degrees(theta):.4f} deg")
