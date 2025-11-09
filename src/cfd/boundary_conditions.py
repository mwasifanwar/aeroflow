import numpy as np

class BoundaryConditions:
    def __init__(self):
        self.config = Config()
    
    def apply_farfield_conditions(self, mesh_points, flow_variables, alpha=5.0, mach=0.0):
        alpha_rad = np.radians(alpha)
        u_inf = np.cos(alpha_rad)
        v_inf = np.sin(alpha_rad)
        
        farfield_indices = self._get_farfield_indices(mesh_points)
        
        flow_variables['velocity_x'][farfield_indices] = u_inf
        flow_variables['velocity_y'][farfield_indices] = v_inf
        flow_variables['pressure'][farfield_indices] = 1.0
        
        return flow_variables
    
    def apply_airfoil_conditions(self, mesh_points, airfoil_points, flow_variables):
        airfoil_indices = self._get_airfoil_indices(mesh_points, airfoil_points)
        
        flow_variables['velocity_x'][airfoil_indices] = 0.0
        flow_variables['velocity_y'][airfoil_indices] = 0.0
        
        return flow_variables
    
    def _get_farfield_indices(self, mesh_points, threshold=15.0):
        distances = np.linalg.norm(mesh_points, axis=1)
        return distances > threshold
    
    def _get_airfoil_indices(self, mesh_points, airfoil_points, tolerance=0.05):
        airfoil_indices = []
        for i, point in enumerate(mesh_points):
            distances = np.linalg.norm(airfoil_points - point, axis=1)
            if np.min(distances) < tolerance:
                airfoil_indices.append(i)
        return airfoil_indices
    
    def apply_symmetry_conditions(self, mesh_points, flow_variables):
        symmetry_indices = self._get_symmetry_indices(mesh_points)
        
        flow_variables['velocity_y'][symmetry_indices] = 0.0
        
        return flow_variables
    
    def _get_symmetry_indices(self, mesh_points, tolerance=0.01):
        return np.where(np.abs(mesh_points[:, 1]) < tolerance)[0]