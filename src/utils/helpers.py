import numpy as np
import torch

def normalize_data(data, mean=None, std=None):
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (data - mean) / std, mean, std

def denormalize_data(data, mean, std):
    return data * std + mean

def calculate_aerodynamic_coefficients(pressure, density, velocity, area):
    dynamic_pressure = 0.5 * density * velocity**2
    lift = np.sum(pressure * area)
    drag = np.sum(pressure * area)
    cl = lift / (dynamic_pressure * area)
    cd = drag / (dynamic_pressure * area)
    return cl, cd

def mesh_grid_to_tensor(grid_points):
    return torch.FloatTensor(grid_points)

def save_simulation_data(geometry, flow_field, coefficients, filename):
    import pickle
    data = {
        'geometry': geometry,
        'flow_field': flow_field,
        'coefficients': coefficients
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_simulation_data(filename):
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)