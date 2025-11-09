import matplotlib.pyplot as plt
import numpy as np

class FlowVisualizer:
    def __init__(self):
        self.config = Config()
    
    def plot_velocity_field(self, mesh_points, velocity_x, velocity_y, airfoil_points=None):
        plt.figure(figsize=(12, 6))
        
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        
        plt.quiver(mesh_points[:, 0], mesh_points[:, 1], velocity_x, velocity_y, speed, 
                  cmap='viridis', scale=20, width=0.002)
        
        if airfoil_points is not None:
            plt.plot(airfoil_points[:, 0], airfoil_points[:, 1], 'r-', linewidth=2)
        
        plt.colorbar(label='Velocity Magnitude')
        plt.axis('equal')
        plt.title('Velocity Field')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    def plot_streamlines(self, mesh_points, velocity_x, velocity_y, airfoil_points=None):
        plt.figure(figsize=(12, 6))
        
        x_unique = np.unique(mesh_points[:, 0])
        y_unique = np.unique(mesh_points[:, 1])
        
        X, Y = np.meshgrid(x_unique, y_unique)
        
        U = velocity_x.reshape(len(y_unique), len(x_unique))
        V = velocity_y.reshape(len(y_unique), len(x_unique))
        
        plt.streamplot(X, Y, U, V, density=2, color='blue', linewidth=1)
        
        if airfoil_points is not None:
            plt.plot(airfoil_points[:, 0], airfoil_points[:, 1], 'r-', linewidth=2)
        
        plt.axis('equal')
        plt.title('Streamlines')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    def plot_pressure_contour(self, mesh_points, pressure, airfoil_points=None):
        plt.figure(figsize=(12, 6))
        
        x_unique = np.unique(mesh_points[:, 0])
        y_unique = np.unique(mesh_points[:, 1])
        
        X, Y = np.meshgrid(x_unique, y_unique)
        P = pressure.reshape(len(y_unique), len(x_unique))
        
        contour = plt.contourf(X, Y, P, levels=50, cmap='jet')
        plt.colorbar(contour, label='Pressure')
        
        if airfoil_points is not None:
            plt.plot(airfoil_points[:, 0], airfoil_points[:, 1], 'k-', linewidth=2)
        
        plt.axis('equal')
        plt.title('Pressure Contour')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()