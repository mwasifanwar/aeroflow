import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.geometry.airfoil_generator import AirfoilGenerator
from src.cfd.solver import CFDSolver
from src.cfd.post_processor import PostProcessor

class TestCFD(unittest.TestCase):
    def test_airfoil_generation(self):
        generator = AirfoilGenerator()
        airfoil = generator.naca_4_digit("0012")
        self.assertEqual(len(airfoil), 400)
    
    def test_cfd_solver(self):
        generator = AirfoilGenerator()
        airfoil = generator.naca_4_digit("0012")
        
        solver = CFDSolver()
        mesh = [[x, y] for x in [-5, 0, 5] for y in [-5, 0, 5]]
        mesh = np.array(mesh)
        
        flow_variables = solver.solve_potential_flow(mesh, airfoil, alpha=5.0)
        self.assertIn('velocity_x', flow_variables)
        self.assertIn('velocity_y', flow_variables)
        self.assertIn('pressure', flow_variables)
    
    def test_aerodynamic_coefficients(self):
        post_processor = PostProcessor()
        flow_variables = {
            'velocity_x': np.array([1, 1, 1]),
            'velocity_y': np.array([0, 0, 0]),
            'pressure': np.array([1, 1, 1])
        }
        airfoil = np.array([[0, 0], [1, 0.1], [0, -0.1]])
        mesh = np.array([[0, 0], [1, 0.1], [0, -0.1]])
        
        coefficients = post_processor.calculate_aerodynamic_coefficients(flow_variables, airfoil, mesh)
        self.assertIn('cl', coefficients)
        self.assertIn('cd', coefficients)
        self.assertIn('cm', coefficients)

if __name__ == '__main__':
    unittest.main()