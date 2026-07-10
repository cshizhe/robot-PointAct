"""
Rotation representation conversions (NumPy).

Supports:
- Euler angles (radians) <-> rotation matrix
- Axis angles <-> rotation matrix
- Quaternion (w, x, y, z) <-> rotation matrix
- Rotation 6D (Zhou et al. 2019: first two columns) <-> rotation matrix
Plus convenience pairwise converters via matrices.

Conventions:
- Euler: intrinsic rotations about axes in the given order string.
  Default order="xyz" (rotate about x, then y, then z in the *rotating* frame).
  If you want common "roll-pitch-yaw" as extrinsic XYZ about fixed axes, that equals intrinsic "zyx".
- Quaternion: (w, x, y, z). Right-handed, active rotations.
"""

import numpy as np


# ---------------------------
# Helpers
# ---------------------------

def _to_numpy(a, dtype=np.float64):
    return np.asarray(a, dtype=dtype)

def _normalize(v, eps=1e-12):
    v = _to_numpy(v)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def _skew(v):
    # v: (..., 3)
    v = _to_numpy(v)
    z = np.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([
        np.stack([z, -vz,  vy], axis=-1),
        np.stack([vz,  z, -vx], axis=-1),
        np.stack([-vy, vx,  z], axis=-1),
    ], axis=-2)

def _axis_angle_to_matrix(axis, angle):
    # Rodrigues
    axis = _normalize(axis)
    angle = _to_numpy(angle)
    K = _skew(axis)
    I = np.eye(3, dtype=axis.dtype)
    # broadcast I to batch
    I = np.broadcast_to(I, K.shape)
    sa = np.sin(angle)[..., None, None]
    ca = np.cos(angle)[..., None, None]
    return I + sa * K + (1.0 - ca) * (K @ K)

def _ensure_matrix_shape(R):
    R = _to_numpy(R)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrix must have shape (...,3,3); got {R.shape}")
    return R

def _wrap_angle(x):
    # wrap to (-pi, pi]
    return (x + np.pi) % (2 * np.pi) - np.pi


# ---------------------------
# Euler <-> Matrix
# ---------------------------

def euler_to_matrix(euler, order: str = "xyz"):
    """
    Convert Euler angles to rotation matrix.

    euler: (..., 3) angles in radians.
    order: axes string like "xyz", "zyx" (intrinsic rotations).

    Returns: (..., 3, 3)
    """
    e = _to_numpy(euler)
    if e.shape[-1] != 3:
        raise ValueError("euler must have shape (..., 3)")
    order = order.lower()
    if order not in {"xyz", "zyx"}:
        raise NotImplementedError("Implemented orders: 'xyz', 'zyx' (intrinsic).")

    ax = {"x": np.array([1.0, 0.0, 0.0], dtype=e.dtype),
          "y": np.array([0.0, 1.0, 0.0], dtype=e.dtype),
          "z": np.array([0.0, 0.0, 1.0], dtype=e.dtype)}

    a0, a1, a2 = e[..., 0], e[..., 1], e[..., 2]
    R0 = _axis_angle_to_matrix(ax[order[0]], a0)
    R1 = _axis_angle_to_matrix(ax[order[1]], a1)
    R2 = _axis_angle_to_matrix(ax[order[2]], a2)
    return R0 @ R1 @ R2

def matrix_to_euler(R, order: str = "xyz", eps: float = 1e-7, wrap: bool = True):
    """
    Convert rotation matrix to Euler angles (intrinsic) for common orders.

    R: (...,3,3)
    order: 'xyz' or 'zyx' (intrinsic).
    eps: singularity threshold.
    wrap: if True, wrap output angles to (-pi, pi].

    Returns: (..., 3) angles in radians.
    """
    R = _ensure_matrix_shape(R)
    order = order.lower()
    if order not in {"xyz", "zyx"}:
        raise NotImplementedError("Implemented orders: 'xyz', 'zyx' (intrinsic).")

    # We derive formulas for intrinsic rotations:
    # xyz intrinsic: R = Rx(a) Ry(b) Rz(c)
    # zyx intrinsic: R = Rz(a) Ry(b) Rx(c)
    if order == "xyz":
        # b from R[0,2] = sin(b)
        sb = np.clip(R[..., 0, 2], -1.0, 1.0)
        b = np.arcsin(sb)
        cb = np.cos(b)
        singular = np.abs(cb) < eps

        a = np.where(
            singular,
            np.arctan2(-R[..., 1, 0], R[..., 1, 1]),  # fallback
            np.arctan2(-R[..., 1, 2], R[..., 2, 2]),
        )
        c = np.where(
            singular,
            0.0,
            np.arctan2(-R[..., 0, 1], R[..., 0, 0]),
        )
        out = np.stack([a, b, c], axis=-1)

    else:  # "zyx"
        # b from R[2,0] = -sin(b)
        sb = np.clip(-R[..., 2, 0], -1.0, 1.0)
        b = np.arcsin(sb)
        cb = np.cos(b)
        singular = np.abs(cb) < eps

        a = np.where(
            singular,
            np.arctan2(-R[..., 0, 1], R[..., 1, 1]),  # fallback
            np.arctan2(R[..., 1, 0], R[..., 0, 0]),
        )
        c = np.where(
            singular,
            0.0,
            np.arctan2(R[..., 2, 1], R[..., 2, 2]),
        )
        out = np.stack([a, b, c], axis=-1)

    return _wrap_angle(out) if wrap else out


# ---------------------------
# Quaternion <-> Matrix
# ---------------------------

def quat_to_matrix(q, normalize: bool = True, order: str = "wxyz"):
    """
    Quaternion -> rotation matrix.

    q: (...,4)
    Returns: (...,3,3)
    """
    if order not in {"wxyz", "xyzw"}:
        raise NotImplementedError("Implemented orders: 'wxyz', 'xyzw'.")

    q = _to_numpy(q)
    if q.shape[-1] != 4:
        raise ValueError(f"q must have shape (..., 4) as {order}")
    if normalize:
        q = _normalize(q)

    if order == "wxyz":
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    elif order == "xyzw":
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R00 = ww + xx - yy - zz
    R01 = 2*(xy - wz)
    R02 = 2*(xz + wy)

    R10 = 2*(xy + wz)
    R11 = ww - xx + yy - zz
    R12 = 2*(yz - wx)

    R20 = 2*(xz - wy)
    R21 = 2*(yz + wx)
    R22 = ww - xx - yy + zz

    return np.stack([
        np.stack([R00, R01, R02], axis=-1),
        np.stack([R10, R11, R12], axis=-1),
        np.stack([R20, R21, R22], axis=-1),
    ], axis=-2)

def matrix_to_quat(R, normalize: bool = True, order: str = "wxyz"):
    """
    Rotation matrix -> quaternion.
    Uses a numerically-stable branch method.

    R: (...,3,3)
    Returns: (...,4)
    """
    if order not in {"wxyz", "xyzw"}:
        raise NotImplementedError("Implemented orders: 'wxyz', 'xyzw'.")

    R = _ensure_matrix_shape(R)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    q = np.empty(R.shape[:-2] + (4,), dtype=R.dtype)

    # Masks
    m0 = tr > 0
    m1 = (R[..., 0, 0] >= R[..., 1, 1]) & (R[..., 0, 0] >= R[..., 2, 2]) & (~m0)
    m2 = (R[..., 1, 1] >  R[..., 2, 2]) & (~m0) & (~m1)
    m3 = (~m0) & (~m1) & (~m2)

    # Case 0: trace positive
    t = np.where(m0, tr + 1.0, 1.0)
    s = 0.5 / np.sqrt(t)
    q[..., 0] = np.where(m0, 0.25 / s, 0.0)
    q[..., 1] = np.where(m0, (R[..., 2, 1] - R[..., 1, 2]) * s, 0.0)
    q[..., 2] = np.where(m0, (R[..., 0, 2] - R[..., 2, 0]) * s, 0.0)
    q[..., 3] = np.where(m0, (R[..., 1, 0] - R[..., 0, 1]) * s, 0.0)

    # Case 1: R00 largest
    t = R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2] + 1.0
    s = 0.5 / np.sqrt(np.maximum(t, 1e-12))
    q[..., 0] = np.where(m1, (R[..., 2, 1] - R[..., 1, 2]) * s, q[..., 0])
    q[..., 1] = np.where(m1, 0.25 / s, q[..., 1])
    q[..., 2] = np.where(m1, (R[..., 0, 1] + R[..., 1, 0]) * s, q[..., 2])
    q[..., 3] = np.where(m1, (R[..., 0, 2] + R[..., 2, 0]) * s, q[..., 3])

    # Case 2: R11 largest
    t = -R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2] + 1.0
    s = 0.5 / np.sqrt(np.maximum(t, 1e-12))
    q[..., 0] = np.where(m2, (R[..., 0, 2] - R[..., 2, 0]) * s, q[..., 0])
    q[..., 1] = np.where(m2, (R[..., 0, 1] + R[..., 1, 0]) * s, q[..., 1])
    q[..., 2] = np.where(m2, 0.25 / s, q[..., 2])
    q[..., 3] = np.where(m2, (R[..., 1, 2] + R[..., 2, 1]) * s, q[..., 3])

    # Case 3: R22 largest
    t = -R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2] + 1.0
    s = 0.5 / np.sqrt(np.maximum(t, 1e-12))
    q[..., 0] = np.where(m3, (R[..., 1, 0] - R[..., 0, 1]) * s, q[..., 0])
    q[..., 1] = np.where(m3, (R[..., 0, 2] + R[..., 2, 0]) * s, q[..., 1])
    q[..., 2] = np.where(m3, (R[..., 1, 2] + R[..., 2, 1]) * s, q[..., 2])
    q[..., 3] = np.where(m3, 0.25 / s, q[..., 3])

    if normalize:
        q = _normalize(q)

    # Optional: enforce a canonical sign (w >= 0) to reduce discontinuities
    sign = np.where(q[..., 0:1] < 0, -1.0, 1.0)
    q = q * sign

    if order == "xyzw":
        q = np.concatenate(
            [q[..., 1:], q[..., :1]], axis=-1
        )
    return q


# ---------------------------
# Rotation 6D <-> Matrix
# ---------------------------

def rot6d_to_matrix(r6d, eps: float = 1e-12):
    """
    Rotation 6D -> rotation matrix.

    r6d: (...,6), interpreted as two 3D vectors (a1,a2) that become the first two
         columns of a rotation matrix after Gram-Schmidt orthonormalization.

    Returns: (...,3,3)
    """
    r6d = _to_numpy(r6d)
    if r6d.shape[-1] != 6:
        raise ValueError("r6d must have shape (..., 6)")
    a1 = r6d[..., 0:3]
    a2 = r6d[..., 3:6]

    b1 = _normalize(a1, eps=eps)
    # make a2 orthogonal to b1
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = _normalize(a2 - dot * b1, eps=eps)
    b3 = np.cross(b1, b2, axis=-1)

    # columns are b1,b2,b3
    return np.stack([b1, b2, b3], axis=-1)

def matrix_to_rot6d(R):
    """
    Rotation matrix -> rotation 6D (first two columns).

    R: (...,3,3)
    Returns: (...,6)
    """
    R = _ensure_matrix_shape(R)
    r6d = np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)
    return r6d

# ---------------------------
# Axis-angle (rotation vector) <-> Matrix
# ---------------------------

def axis_angle_to_matrix_3(vec, eps: float = 1e-12):
    """
    Axis-angle (rotation vector) -> rotation matrix.
    vec: (...,3) rotation vector where ||vec|| = angle and direction is axis.
    Returns: (...,3,3)
    """
    vec = _to_numpy(vec)
    if vec.shape[-1] != 3:
        raise ValueError("axis-angle vector must have shape (...,3)")

    angle = np.linalg.norm(vec, axis=-1)
    axis = np.where(angle[..., None] > eps, vec / angle[..., None], np.array([1.0,0.0,0.0]))
    angle = np.maximum(angle, 0.0)
    return _axis_angle_to_matrix(axis, angle)


def matrix_to_axis_angle_3(R, eps: float = 1e-7):
    """
    Rotation matrix -> axis-angle rotation vector (Rodrigues).
    Returns: (...,3) vec such that ||vec|| = angle and direction = axis.
    """
    R = _ensure_matrix_shape(R)

    # angle from trace
    tr = R[..., 0,0] + R[..., 1,1] + R[..., 2,2]
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # axis from antisymmetric part
    wx = R[..., 2,1] - R[..., 1,2]
    wy = R[..., 0,2] - R[..., 2,0]
    wz = R[..., 1,0] - R[..., 0,1]
    w = np.stack([wx, wy, wz], axis=-1)

    sin_theta2 = np.linalg.norm(w, axis=-1)
    mask = sin_theta2 > eps

    axis = np.zeros_like(w)
    denom = np.maximum(sin_theta2[..., None], eps)
    axis = np.where(mask[..., None], w / denom, axis)

    # default axis for small angle
    axis[...,0] = np.where(mask, axis[...,0], 1.0)

    axis = _normalize(axis)

    # rotation vector
    return axis * theta[..., None]


# ---------------------------
# Pairwise conversion (via matrix)
# ---------------------------

def to_matrix(x, src: str, *, euler_order: str = "xyz", quat_order: str = "wxyz"):
    """
    Convert from a representation to rotation matrix.

    src in {"matrix","euler","quat","rot6d"}.
    - euler: (...,3) radians
    - quat: (...,4)
    - rot6d: (...,6)
    - matrix: (...,3,3)
    """
    src = src.lower()
    if src == "matrix":
        return _ensure_matrix_shape(x)
    if src == "euler":
        return euler_to_matrix(x, order=euler_order)
    if src == "quat":
        return quat_to_matrix(x, order=quat_order)
    if src == "rot6d":
        return rot6d_to_matrix(x)
    if src == "axisangle":
        return axis_angle_to_matrix_3(x)
    raise ValueError(f"Unknown src '{src}'")

def from_matrix(R, dst: str, *, euler_order: str = "xyz", quat_order: str = "wxyz"):
    """
    Convert rotation matrix to a representation.

    dst in {"matrix","euler","quat","rot6d"}.
    """
    dst = dst.lower()
    if dst == "matrix":
        return _ensure_matrix_shape(R)
    if dst == "euler":
        return matrix_to_euler(R, order=euler_order)
    if dst == "quat":
        return matrix_to_quat(R, order=quat_order)
    if dst == "rot6d":
        return matrix_to_rot6d(R)
    if dst == "axisangle":
        return matrix_to_axis_angle_3(R)
    raise ValueError(f"Unknown dst '{dst}'")

def convert_rotation(
    x, src: str, dst: str, *, 
    euler_order_src: str = "xyz", euler_order_dst: str = "xyz",
    quat_order_src: str = "wxyz", quat_order_dst: str = "wxyz"
):
    """
    Generic pairwise converter between representations.

    src in {"matrix","euler","quat","rot6d","axisangle"}
    dst in {"matrix","euler","quat","rot6d","axisangle"}

    Example:
        q = convert_rotation(euler, "euler", "quat", euler_order_src="zyx")
        e = convert_rotation(q, "quat", "euler", euler_order_dst="xyz")

    Args:
        - x: input rotation in src representation. shape (..., D)
    """
    R = to_matrix(x, src, euler_order=euler_order_src, quat_order=quat_order_src)
    return from_matrix(R, dst, euler_order=euler_order_dst, quat_order=quat_order_dst)


# ---------------------------
# Quick sanity check (optional)
# ---------------------------

if __name__ == "__main__":
    # random quaternion -> matrix -> quaternion roundtrip
    rng = np.random.default_rng(0)
    q = rng.normal(size=(10, 4))
    q = _normalize(q)
    for i in range(len(q)):
        if q[i, -1] < 0:
            q[i] = -q[i]

    R = convert_rotation(q, "quat", "matrix", quat_order_src="xyzw")
    q2 = convert_rotation(R, "matrix", "quat", quat_order_dst="xyzw")
    # compare up to sign: enforce canonical w>=0 in matrix_to_quat, so q and q2 may differ
    print("quat -> matrix", np.max(np.abs(q - q2)))
    print("quat -> matrix", np.max(np.abs(R - quat_to_matrix(q2, order="xyzw"))))

    # quat -> euler -> quat
    e = convert_rotation(q, "quat", "euler", euler_order_dst="xyz", quat_order_src="xyzw")
    q2 = convert_rotation(e, "euler", "quat", euler_order_src="xyz", quat_order_dst="xyzw")
    print("quat -> euler", np.max(np.abs(q - q2)))

    # quat -> rot6d -> quat
    r = convert_rotation(q, "quat", "rot6d", quat_order_src="xyzw")
    q2 = convert_rotation(r, "rot6d", "quat", quat_order_dst="xyzw")
    print("quat -> rot6d", np.max(np.abs(q - q2)))

    # euler -> rot6d -> euler
    r = convert_rotation(e, "euler", "rot6d", euler_order_src="xyz")
    e2 = convert_rotation(r, "rot6d", "euler", euler_order_dst="xyz")
    print("euler -> rot6d", np.max(np.abs(q - q2)))
