import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualization:
    def __init__(self):
        self.config = Config()
    
    def plot_airfoil(self, airfoil_points, title="Airfoil Geometry"):
        plt.figure(figsize=(10, 4))
        plt.plot(airfoil_points[:, 0], airfoil_points[:, 1], 'b-', linewidth=2)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.show()
    
    def plot_pressure_distribution(self, upper_cp, lower_cp, title="Pressure Distribution"):
        plt.figure(figsize=(8, 6))
        
        if len(upper_cp[0]) > 0:
            plt.plot(upper_cp[0], upper_cp[1], 'r-', linewidth=2, label='Upper Surface')
        if len(lower_cp[0]) > 0:
            plt.plot(lower_cp[0], lower_cp[1], 'b-', linewidth=2, label='Lower Surface')
        
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.xlabel('x/c')
        plt.ylabel('Cp')
        plt.legend()
        plt.show()
    
    def plot_optimization_history(self, fitness_history, title="Optimization Progress"):
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, 'b-', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.yscale('log')
        plt.show()
    
    def create_interactive_airfoil_plot(self, airfoil_points, flow_variables=None, mesh_points=None):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Airfoil Geometry', 'Pressure Distribution'))
        
        fig.add_trace(
            go.Scatter(x=airfoil_points[:, 0], y=airfoil_points[:, 1], 
                      mode='lines', name='Airfoil'),
            row=1, col=1
        )
        
        if flow_variables is not None and mesh_points is not None:
            fig.add_trace(
                go.Scatter(x=mesh_points[:, 0], y=mesh_points[:, 1],
                          mode='markers', marker=dict(
                              color=flow_variables['pressure'],
                              colorscale='Viridis',
                              size=3,
                              showscale=True
                          ), name='Pressure'),
                row=1, col=2
            )
        
        fig.update_layout(height=500, width=1000, title_text="AeroFlow Analysis")
        fig.show()
    
    def plot_mesh(self, mesh_points, airfoil_points=None):
        plt.figure(figsize=(12, 6))
        plt.scatter(mesh_points[:, 0], mesh_points[:, 1], c='gray', s=1, alpha=0.6)
        
        if airfoil_points is not None:
            plt.plot(airfoil_points[:, 0], airfoil_points[:, 1], 'r-', linewidth=2, label='Airfoil')
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title('Computational Mesh')
        plt.xlabel('x')
        plt.ylabel('y')
        if airfoil_points is not None:
            plt.legend()
        plt.show()