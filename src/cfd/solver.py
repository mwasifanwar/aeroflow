import numpy as np
import torch

class CFDSolver:
    def __init__(self):
        self.config = Config()
    
    def solve_potential_flow(self, mesh_points, airfoil_points, alpha=5.0, mach=0.0):
        reynolds = self.config.get('cfd.reynolds_number')
        
        u = np.ones_like(mesh_points[:, 0])
        v = np.zeros_like(mesh_points[:, 1])
        p = np.ones_like(mesh_points[:, 0])
        
        alpha_rad = np.radians(alpha)
        
        for iteration in range(self.config.get('cfd.max_iterations')):
            u_old = u.copy()
            v_old = v.copy()
            
            for i in range(len(mesh_points)):
                if self._is_airfoil_point(mesh_points[i], airfoil_points):
                    u[i] = 0
                    v[i] = 0
                    continue
                
                dx = 0.1
                dy = 0.1
                
                laplacian_u = self._laplacian(u, mesh_points, i, dx, dy)
                laplacian_v = self._laplacian(v, mesh_points, i, dx, dy)
                
                u[i] = u_old[i] + 0.1 * laplacian_u
                v[i] = v_old[i] + 0.1 * laplacian_v
            
            convergence = np.max(np.abs(u - u_old)) + np.max(np.abs(v - v_old))
            if convergence < self.config.get('cfd.convergence_tolerance'):
                break
        
        p = 1 - 0.5 * (u**2 + v**2)
        
        return {'velocity_x': u, 'velocity_y': v, 'pressure': p}
    
    def _is_airfoil_point(self, point, airfoil_points, tolerance=0.01):
        distances = np.linalg.norm(airfoil_points - point, axis=1)
        return np.min(distances) < tolerance
    
    def _laplacian(self, field, points, index, dx, dy):
        i, j = self._get_grid_index(points, index)
        if i is None:
            return 0
        
        laplacian = 0
        count = 0
        
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            neighbor_index = self._get_point_index(points, ni, nj)
            if neighbor_index is not None:
                laplacian += field[neighbor_index] - field[index]
                count += 1
        
        return laplacian / max(count, 1)
    
    def _get_grid_index(self, points, index):
        x_unique = np.unique(points[:, 0])
        y_unique = np.unique(points[:, 1])
        
        x_idx = np.where(np.isclose(x_unique, points[index, 0]))[0]
        y_idx = np.where(np.isclose(y_unique, points[index, 1]))[0]
        
        if len(x_idx) > 0 and len(y_idx) > 0:
            return x_idx[0], y_idx[0]
        return None, None
    
    def _get_point_index(self, points, i, j):
        x_unique = np.unique(points[:, 0])
        y_unique = np.unique(points[:, 1])
        
        if 0 <= i < len(x_unique) and 0 <= j < len(y_unique):
            x = x_unique[i]
            y = y_unique[j]
            distances = np.linalg.norm(points - [x, y], axis=1)
            return np.argmin(distances)
        return None
    
    def solve_navier_stokes_simplified(self, mesh_points, airfoil_points, alpha=5.0, reynolds=1e6):
        u = np.ones(len(mesh_points))
        v = np.zeros(len(mesh_points))
        p = np.ones(len(mesh_points))
        
        dt = 0.01
        nu = 1.0 / reynolds
        
        for iteration in range(100):
            u_old = u.copy()
            v_old = v.copy()
            
            for i in range(len(mesh_points)):
                if self._is_airfoil_point(mesh_points[i], airfoil_points):
                    u[i] = 0
                    v[i] = 0
                    continue
                
                convection = u[i] * self._gradient(u, mesh_points, i, 0) + v[i] * self._gradient(u, mesh_points, i, 1)
                diffusion = nu * self._laplacian(u, mesh_points, i, 0.1, 0.1)
                pressure_grad = self._gradient(p, mesh_points, i, 0)
                
                u[i] = u_old[i] + dt * (-convection + diffusion - pressure_grad)
                v[i] = v_old[i] + dt * (-convection + diffusion - self._gradient(p, mesh_points, i, 1))
            
            divergence = self._calculate_divergence(u, v, mesh_points)
            p = p - 0.1 * divergence
        
        return {'velocity_x': u, 'velocity_y': v, 'pressure': p}
    
    def _gradient(self, field, points, index, direction):
        i, j = self._get_grid_index(points, index)
        if i is None:
            return 0
        
        if direction == 0:
            left_index = self._get_point_index(points, i-1, j)
            right_index = self._get_point_index(points, i+1, j)
            if left_index is not None and right_index is not None:
                return (field[right_index] - field[left_index]) / 0.2
        else:
            bottom_index = self._get_point_index(points, i, j-1)
            top_index = self._get_point_index(points, i, j+1)
            if bottom_index is not None and top_index is not None:
                return (field[top_index] - field[bottom_index]) / 0.2
        
        return 0
    
    def _calculate_divergence(self, u, v, points):
        divergence = np.zeros_like(u)
        for i in range(len(points)):
            du_dx = self._gradient(u, points, i, 0)
            dv_dy = self._gradient(v, points, i, 1)
            divergence[i] = du_dx + dv_dy
        return divergence