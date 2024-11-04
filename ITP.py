# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:04:55 2024

@author: lzj
"""

import numpy as np
import pandas as pd

def distance_weighted_least_squares(gps_points, target_point, p=2, regularization=1e-6):
    """
    Least-squares strain analysis based on distance weighting.
    
    :param gps_points: List of tuples [(x1, y1, v1), (x2, y2, v2), ...] where (xi, yi) are coordinates and vi is the velocity.
    :param target_point: Tuple (x, y) representing the coordinates of the target point.
    :param p: Power parameter for distance weighting (default is 2).
    :param regularization: Regularization parameter to avoid singular matrix (default is 1e-6).
    :return: Interpolated velocity at the target point.
    """
    x, y = target_point
    distances = []
    velocities = []
    
    for (xi, yi, vi) in gps_points:
        distance = np.sqrt((x - xi) ** 2 + (y - yi) ** 2)
        if distance == 0:
            return vi  # If the target point coincides with a known point
        distances.append(distance)
        velocities.append(vi)
    
    weights = np.array([1 / (d ** p) for d in distances])
    
    A = np.array([[1, xi, yi] for (xi, yi, vi) in gps_points])
    b = np.array(velocities)
    
    W = np.diag(weights)
    
    # Solve the weighted least squares problem with regularization
    ATW = np.dot(A.T, W)
    ATA = np.dot(ATW, A) + regularization * np.eye(A.shape[1])
    ATb = np.dot(ATW, b)
    
    coeffs = np.linalg.solve(ATA, ATb)
    
    # Interpolated velocity at the target point
    interpolated_velocity = coeffs[0] + coeffs[1] * x + coeffs[2] * y
    
    return interpolated_velocity

# 读取已知点数据
known_points_df = pd.read_csv('known_points.txt', delim_whitespace=True)
gps_points = known_points_df.to_records(index=False)

# 读取目标点数据
target_points_df = pd.read_csv('target_points.txt', delim_whitespace=True)
target_points = target_points_df.to_records(index=False)

# 对每个目标点进行插值计算
results = []
for target_point in target_points:
    interpolated_velocity = distance_weighted_least_squares(gps_points, (target_point.x, target_point.y))
    results.append((target_point.x, target_point.y, interpolated_velocity))

# 将结果保存到CSV文件
results_df = pd.DataFrame(results, columns=['x', 'y', 'interpolated_velocity'])
results_df.to_csv('interpolated_velocities.csv', index=False)

print("Interpolation completed. Results saved to 'interpolated_velocities.csv'.")

