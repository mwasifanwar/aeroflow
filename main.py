import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.geometry.airfoil_generator import AirfoilGenerator
from src.geometry.mesh_generator import MeshGenerator
from src.cfd.solver import CFDSolver
from src.cfd.post_processor import PostProcessor
from src.neural_networks.surrogate_models import AerodynamicSurrogate
from src.optimization.genetic_algorithm import GeneticOptimizer
from src.visualization.plotter import Visualization
from src.api.server import app
import uvicorn

def run_demo():
    print("Running AeroFlow Demo...")
    
    generator = AirfoilGenerator()
    airfoil = generator.naca_4_digit("2412")
    
    mesh_gen = MeshGenerator()
    mesh = mesh_gen.generate_structured_mesh(airfoil)
    
    solver = CFDSolver()
    flow_variables = solver.solve_potential_flow(mesh, airfoil, alpha=5.0)
    
    post_processor = PostProcessor()
    coefficients = post_processor.calculate_aerodynamic_coefficients(flow_variables, airfoil, mesh, alpha=5.0)
    
    print(f"Aerodynamic Coefficients: CL={coefficients['cl']:.4f}, CD={coefficients['cd']:.4f}, CM={coefficients['cm']:.4f}")
    
    viz = Visualization()
    viz.plot_airfoil(airfoil, "NACA 2412 Airfoil")
    
    return coefficients

def run_api():
    from src.utils.config import Config
    config = Config()
    print(f"Starting AeroFlow API server on {config.get('api.host')}:{config.get('api.port')}")
    uvicorn.run(app, host=config.get('api.host'), port=config.get('api.port'))

def main():
    parser = argparse.ArgumentParser(description='AeroFlow: AI-Driven Aerodynamic Optimization')
    parser.add_argument('--mode', choices=['api', 'demo', 'optimize'], default='demo', help='Operation mode')
    parser.add_argument('--airfoil', type=str, help='NACA digits for airfoil generation')
    parser.add_argument('--alpha', type=float, default=5.0, help='Angle of attack in degrees')
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        run_api()
    elif args.mode == 'demo':
        run_demo()
    elif args.mode == 'optimize':
        print("Running optimization demo...")
        
        surrogate = AerodynamicSurrogate()
        
        design_params = [np.random.randn(10) for _ in range(100)]
        angles = np.random.uniform(-5, 15, 100)
        coefficients = [[np.random.uniform(0.1, 1.5), np.random.uniform(0.01, 0.1), 0.0] for _ in range(100)]
        
        surrogate.train(design_params, angles, coefficients, epochs=500)
        
        bounds = [(-1.0, 1.0) for _ in range(10)]
        optimizer = GeneticOptimizer()
        best_params, best_fitness = optimizer.optimize(surrogate, bounds, alpha=5.0)
        
        print(f"Optimization completed - mwasifanwar")
        print(f"Best parameters: {best_params}")
        print(f"Best fitness: {best_fitness}")
    else:
        print("AeroFlow system ready - mwasifanwar")

if __name__ == "__main__":
    main()