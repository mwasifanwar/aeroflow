import numpy as np

class AirfoilGenerator:
    def __init__(self):
        self.config = Config()
    
    def naca_4_digit(self, digits, points=None):
        if points is None:
            points = self.config.get('geometry.airfoil_points')
        
        m = int(digits[0]) / 100.0
        p = int(digits[1]) / 10.0
        t = int(digits[2:]) / 100.0
        
        x = np.linspace(0, 1, points)
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        
        if p > 0:
            yc = np.where(x < p, 
                         m / p**2 * (2 * p * x - x**2),
                         m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
            dyc_dx = np.where(x < p,
                            2 * m / p**2 * (p - x),
                            2 * m / (1 - p)**2 * (p - x))
            theta = np.arctan(dyc_dx)
            
            xu = x - yt * np.sin(theta)
            yu = yc + yt * np.cos(theta)
            xl = x + yt * np.sin(theta)
            yl = yc - yt * np.cos(theta)
        else:
            xu = x
            yu = yt
            xl = x
            yl = -yt
        
        upper = np.column_stack([xu, yu])
        lower = np.column_stack([xl, yl])
        
        return np.vstack([upper, lower[::-1]])
    
    def bezier_airfoil(self, control_points, points=None):
        if points is None:
            points = self.config.get('geometry.airfoil_points')
        
        def bezier_curve(t, points):
            n = len(points) - 1
            curve = np.zeros((len(t), 2))
            for i in range(n + 1):
                binom = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
                curve += binom * (1 - t)**(n - i) * t**i * points[i]
            return curve
        
        t = np.linspace(0, 1, points)
        upper_curve = bezier_curve(t, control_points[:4])
        lower_curve = bezier_curve(t, control_points[4:])
        
        return np.vstack([upper_curve, lower_curve[::-1]])
    
    def parametric_airfoil(self, parameters, points=None):
        if points is None:
            points = self.config.get('geometry.airfoil_points')
        
        x = np.linspace(0, 1, points)
        
        camber = parameters[0] * (1 - np.cos(np.pi * x))
        thickness = parameters[1] * (1 - x)**0.5
        
        for i in range(2, len(parameters)):
            camber += parameters[i] * np.sin((i-1) * np.pi * x)
            thickness += parameters[i] * np.cos((i-1) * np.pi * x)
        
        upper = np.column_stack([x, camber + thickness/2])
        lower = np.column_stack([x, camber - thickness/2])
        
        return np.vstack([upper, lower[::-1]])