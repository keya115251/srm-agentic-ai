"""
Physics features for the Star surrogate, kept in their own module so the
saved model pipeline can be unpickled by any caller (train_star_surrogate.py
saves a pipeline that calls add_physics_features on the 7 raw inputs).
"""
import numpy as np

PHYSICS_FEATURES = ['At', 'port0', 'perim0', 'endArea', 'Ab0', 'Kn0', 'propVol',
                    'portThroat', 'expRatio', 'tipWeb', 'valleyWeb', 'valleyR']


def star_polygon(num_points, point_length, point_width):
    """Initial port of motorlib's Star Grain as a 2n-vertex polygon: tips at
    radius point_length and valleys where the facing edges of neighbouring
    points meet on the bisector between them. Each point is a triangle with
    its base (width point_width) across the axis, so near the axis the
    points overlap and the port is exactly this polygon for n >= 3.
    Returns (valley radius, port area, perimeter)."""
    n = np.maximum(num_points, 3)
    half = np.pi / n          # angle from a point's axis to the bisector
    # Edge from the base corner (w/2 at 90 deg from the point axis) to the
    # tip (L at 0 deg). In the point's frame: p(s) = (s L, (1 - s) w/2).
    # It meets the bisector (angle `half`) where (1 - s) w/2 = s L tan(half).
    w2 = point_width / 2
    s = w2 / (w2 + point_length * np.tan(half))
    valley_r = s * point_length / np.cos(half)
    # Each of the 2n triangles (axis, tip, valley) has area
    # 0.5 * L * r_v * sin(half); each edge runs tip -> valley.
    area = 2 * n * 0.5 * point_length * valley_r * np.sin(half)
    edge = np.sqrt(point_length ** 2 + valley_r ** 2
                   - 2 * point_length * valley_r * np.cos(half))
    return valley_r, area, 2 * n * edge


def add_physics_features(X):
    """Appends initial-geometry quantities computed from the 7 raw inputs
    (columns in FEATURES order): throat area, initial port area, burning
    perimeter and burning area, initial Kn, propellant volume, port/throat
    ratio, expansion ratio, the web outside the point tips and outside the
    valleys, and the valley radius. Same idea as finocyl_features.py: hand
    the tree model the products and ratios (like Ab0/At) it would otherwise
    approximate with many small splits. Lives inside the saved pipeline, so
    callers still pass the 7 raw inputs.
    """
    X = np.asarray(X, dtype=float)
    diameter, length, num_points, point_length, point_width, throat, exit_dia = X.T
    at = np.pi / 4 * throat ** 2
    valley_r, port0, perim0 = star_polygon(num_points, point_length, point_width)
    end_area = np.pi / 4 * diameter ** 2 - port0
    ab0 = perim0 * length + 2 * end_area   # ends uninhibited, as in the wrapper
    extra = np.column_stack([
        at, port0, perim0, end_area, ab0, ab0 / at, end_area * length,
        port0 / at, (exit_dia / throat) ** 2, diameter / 2 - point_length,
        diameter / 2 - valley_r, valley_r,
    ])
    return np.hstack([X, extra])
