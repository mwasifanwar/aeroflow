import numpy as np

class PostProcessor:
    def __init__(self):
        self.config = Config()
    
    def calculate_aerodynamic_coefficients(self, flow_variables, airfoil_points, mesh_points, alpha=5.0, chord=1.0):
        airfoil_indices = self._get_airfoil_indices(mesh_points, airfoil_points)
        
        if len(airfoil_indices) == 0:
            return {'cl': 0.0, 'cd': 0.0, 'cm': 0.0}
        
        pressure_airfoil = flow_variables['pressure'][airfoil_indices]
        points_airfoil = mesh_points[airfoil_indices]
        
        upper_indices = points_airfoil[:, 1] > 0
        lower_indices = points_airfoil[:, 1] <= 0
        
        if np.sum(upper_indices) == 0 or np.sum(lower_indices) == 0:
            return {'cl': 0.0, 'cd': 0.0, 'cm': 0.0}
        
        upper_pressure = pressure_airfoil[upper_indices]
        lower_pressure = pressure_airfoil[lower_indices]
        upper_points = points_airfoil[upper_indices]
        lower_points = points_airfoil[lower_indices]
        
        upper_points_sorted = upper_points[np.argsort(upper_points[:, 0])]
        lower_points_sorted = lower_points[np.argsort(lower_points[:, 0])]
        
        cp_upper = upper_pressure - 1.0
        cp_lower = lower_pressure - 1.0
        
        dx_upper = np.diff(upper_points_sorted[:, 0])
        dy_upper = np.diff(upper_points_sorted[:, 1])
        
        dx_lower = np.diff(lower_points_sorted[:, 0])
        dy_lower = np.diff(lower_points_sorted[:, 1])
        
        normal_upper = np.column_stack([-dy_upper, dx_upper])
        normal_lower = np.column_stack([dy_lower, -dx_lower])
        
        cp_upper_avg = (cp_upper[:-1] + cp_upper[1:]) / 2
        cp_lower_avg = (cp_lower[:-1] + cp_lower[1:]) / 2
        
        force_upper = np.sum(cp_upper_avg[:, None] * normal_upper, axis=0)
        force_lower = np.sum(cp_lower_avg[:, None] * normal_lower, axis=0)
        
        total_force = force_upper + force_lower
        
        alpha_rad = np.radians(alpha)
        lift = -total_force[0] * np.sin(alpha_rad) + total_force[1] * np.cos(alpha_rad)
        drag = total_force[0] * np.cos(alpha_rad) + total_force[1] * np.sin(alpha_rad)
        
        dynamic_pressure = 0.5 * 1.225 * (self.config.get('cfd.mach_number') * 340.29)**2
        area = chord * 1.0
        
        cl = lift / (dynamic_pressure * area)
        cd = drag / (dynamic_pressure * area)
        
        moment = self._calculate_moment(pressure_airfoil, points_airfoil, chord)
        cm = moment / (dynamic_pressure * area * chord)
        
        return {'cl': cl, 'cd': cd, 'cm': cm}
    
    def _get_airfoil_indices(self, mesh_points, airfoil_points, tolerance=0.05):
        airfoil_indices = []
        for i, point in enumerate(mesh_points):
            distances = np.linalg.norm(airfoil_points - point, axis=1)
            if np.min(distances) < tolerance:
                airfoil_indices.append(i)
        return airfoil_indices
    
    def _calculate_moment(self, pressure, points, chord):
        center = np.array([0.25 * chord, 0])
        moments = []
        
        for i in range(len(pressure)-1):
            point1 = points[i]
            point2 = points[i+1]
            midpoint = (point1 + point2) / 2
            force = (pressure[i] + pressure[i+1]) / 2
            lever_arm = midpoint - center
            moment = force * lever_arm[0]
            moments.append(moment)
        
        return np.sum(moments) if moments else 0.0
    
    def calculate_pressure_distribution(self, flow_variables, airfoil_points, mesh_points):
        airfoil_indices = self._get_airfoil_indices(mesh_points, airfoil_points)
        
        if len(airfoil_indices) == 0:
            return np.array([]), np.array([])
        
        pressure_airfoil = flow_variables['pressure'][airfoil_indices]
        points_airfoil = mesh_points[airfoil_indices]
        
        upper_indices = points_airfoil[:, 1] > 0
        lower_indices = points_airfoil[:, 1] <= 0
        
        upper_x = points_airfoil[upper_indices, 0]
        upper_cp = pressure_airfoil[upper_indices] - 1.0
        
        lower_x = points_airfoil[lower_indices, 0]
        lower_cp = pressure_airfoil[lower_indices] - 1.0
        
        upper_sorted = upper_x[np.argsort(upper_x)], upper_cp[np.argsort(upper_x)]
        lower_sorted = lower_x[np.argsort(lower_x)], lower_cp[np.argsort(lower_x)]
        
        return upper_sorted, lower_sorted