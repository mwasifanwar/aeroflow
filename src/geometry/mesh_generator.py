import numpy as np

class MeshGenerator:
    def __init__(self):
        self.config = Config()
    
    def generate_structured_mesh(self, airfoil_points, domain_size=10.0, resolution=None):
        if resolution is None:
            resolution = self.config.get('cfd.mesh_resolution')
        
        nx, ny = resolution
        
        airfoil_x = airfoil_points[:, 0]
        airfoil_y = airfoil_points[:, 1]
        
        x_min, x_max = -domain_size/4, domain_size
        y_min, y_max = -domain_size/2, domain_size/2
        
        xi = np.linspace(0, 1, nx)
        eta = np.linspace(0, 1, ny)
        
        mesh_points = []
        for i in range(ny):
            for j in range(nx):
                if i < ny//2:
                    r = eta[i] * 2
                    x = x_min + (x_max - x_min) * xi[j]
                    y = y_min + (0 - y_min) * r
                else:
                    r = (eta[i] - 0.5) * 2
                    x = x_min + (x_max - x_min) * xi[j]
                    y = 0 + (y_max - 0) * r
                
                mesh_points.append([x, y])
        
        return np.array(mesh_points)
    
    def create_c_grid(self, airfoil_points, farfield_radius=20.0, points_radial=50, points_angular=100):
        theta = np.linspace(0, 2*np.pi, points_angular)
        
        mesh_points = []
        for r_frac in np.linspace(0, 1, points_radial):
            radius = 1 + r_frac * (farfield_radius - 1)
            
            for angle in theta:
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                mesh_points.append([x, y])
        
        return np.array(mesh_points)
    
    def refine_mesh_near_airfoil(self, base_mesh, airfoil_points, refinement_radius=2.0, refinement_factor=2.0):
        airfoil_center = np.mean(airfoil_points, axis=0)
        
        refined_mesh = []
        for point in base_mesh:
            distance = np.linalg.norm(point - airfoil_center)
            if distance < refinement_radius:
                for _ in range(int(refinement_factor)):
                    noise = np.random.normal(0, 0.01, 2)
                    refined_mesh.append(point + noise)
            else:
                refined_mesh.append(point)
        
        return np.array(refined_mesh)