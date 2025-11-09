import numpy as np
from scipy.stats import norm

class BayesianOptimizer:
    def __init__(self):
        self.config = Config()
        self.observed_points = []
        self.observed_values = []
    
    def optimize(self, objective_function, bounds, n_iter=50):
        x0 = self._random_sample(bounds)
        self.observed_points.append(x0)
        self.observed_values.append(objective_function(x0))
        
        for i in range(n_iter):
            x_next = self._acquisition_function(bounds)
            y_next = objective_function(x_next)
            
            self.observed_points.append(x_next)
            self.observed_values.append(y_next)
            
            if i % 5 == 0:
                best_idx = np.argmin(self.observed_values)
                print(f"mwasifanwar Iteration {i}, Best: {self.observed_values[best_idx]:.6f}")
        
        best_idx = np.argmin(self.observed_values)
        return self.observed_points[best_idx], self.observed_values[best_idx]
    
    def _random_sample(self, bounds):
        return np.array([np.random.uniform(low, high) for low, high in bounds])
    
    def _acquisition_function(self, bounds, xi=0.01):
        best_value = min(self.observed_values)
        
        def expected_improvement(x):
            x = np.array(x).reshape(1, -1)
            mu, sigma = self._surrogate_model(x)
            
            if sigma == 0:
                return 0
            
            Z = (best_value - mu - xi) / sigma
            return (best_value - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        best_x = None
        best_ei = -float('inf')
        
        for _ in range(1000):
            x_candidate = self._random_sample(bounds)
            ei = expected_improvement(x_candidate)
            
            if ei > best_ei:
                best_ei = ei
                best_x = x_candidate
        
        return best_x
    
    def _surrogate_model(self, x):
        if len(self.observed_points) < 2:
            return np.mean(self.observed_values), 1.0
        
        distances = [np.linalg.norm(x - point) for point in self.observed_points]
        weights = 1.0 / (np.array(distances) + 1e-8)
        weights = weights / np.sum(weights)
        
        mu = np.sum(weights * self.observed_values)
        
        variance = np.sum(weights * (np.array(self.observed_values) - mu)**2)
        sigma = np.sqrt(variance)
        
        return mu, sigma