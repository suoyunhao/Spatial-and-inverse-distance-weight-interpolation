This is a small procedure that satisfies the spatial interpolation of the velocity field.
    
    Include:
    :param gps_points: List of tuples [(x1, y1, v1), (x2, y2, v2), ...] where (xi, yi) are coordinates and vi is the velocity.
    :param target_point: Tuple (x, y) representing the coordinates of the target point.
    :param p: Power parameter for distance weighting (default is 2).
    :param regularization: Regularization parameter to avoid singular matrix (default is 1e-6).
    :return: Interpolated velocity at the target point.
