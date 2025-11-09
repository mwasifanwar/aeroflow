import numpy as np

class Parameterization:
    def __init__(self):
        self.config = Config()
    
    def hicks_henne_bumps(self, parameters, x):
        bumps = np.zeros_like(x)
        for i, param in enumerate(parameters):
            location = (i + 1) / (len(parameters) + 1)
            width = 0.1
            bumps += param * np.sin(np.pi * x**np.log(0.5) / np.log(location))**4
        return bumps
    
    def cst_parameterization(self, parameters, x, class_function=True):
        n = len(parameters) - 1
        shape_function = np.zeros_like(x)
        
        for i in range(n + 1):
            binom = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
            shape_function += parameters[i] * binom * x**i * (1 - x)**(n - i)
        
        if class_function:
            class_func = x**0.5 * (1 - x)
            shape_function *= class_func
        
        return shape_function
    
    def get_design_variables(self, airfoil_points, method='cst', n_params=None):
        if n_params is None:
            n_params = self.config.get('geometry.parameter_dim')
        
        x = airfoil_points[:len(airfoil_points)//2, 0]
        y_upper = airfoil_points[:len(airfoil_points)//2, 1]
        y_lower = airfoil_points[len(airfoil_points)//2:, 1]
        
        if method == 'cst':
            class_func = x**0.5 * (1 - x)
            upper_params = np.polyfit(x, y_upper/class_func, n_params-1)
            lower_params = np.polyfit(x, y_lower/class_func, n_params-1)
            return np.concatenate([upper_params, lower_params])
        
        elif method == 'coordinates':
            return airfoil_points.flatten()
    
    def reconstruct_airfoil(self, parameters, method='cst', points=None):
        if points is None:
            points = self.config.get('geometry.airfoil_points')
        
        x = np.linspace(0, 1, points)
        
        if method == 'cst':
            n = len(parameters) // 2
            upper_params = parameters[:n]
            lower_params = parameters[n:]
            
            y_upper = self.cst_parameterization(upper_params, x)
            y_lower = self.cst_parameterization(lower_params, x)
            
            upper = np.column_stack([x, y_upper])
            lower = np.column_stack([x, y_lower])
            
            return np.vstack([upper, lower[::-1]])
        
        elif method == 'coordinates':
            return parameters.reshape(-1, 2)