import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.neural_networks.surrogate_models import AerodynamicSurrogate

class TestNeuralNetworks(unittest.TestCase):
    def test_surrogate_model(self):
        surrogate = AerodynamicSurrogate()
        
        design_params = [np.random.randn(10) for _ in range(10)]
        angles = np.random.uniform(-5, 15, 10)
        coefficients = [[0.5, 0.01, 0.0] for _ in range(10)]
        
        surrogate.train(design_params, angles, coefficients, epochs=10)
        
        test_params = np.random.randn(10)
        prediction = surrogate.predict(test_params, 5.0)
        
        self.assertIn('cl', prediction)
        self.assertIn('cd', prediction)
        self.assertIn('cm', prediction)

if __name__ == '__main__':
    unittest.main()