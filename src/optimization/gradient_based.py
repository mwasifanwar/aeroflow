import torch
import numpy as np

class GradientOptimizer:
    def __init__(self):
        self.config = Config()
    
    def optimize(self, surrogate_model, initial_params, alpha=5.0, learning_rate=0.01, iterations=100):
        params = torch.tensor(initial_params, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=learning_rate)
        
        best_params = initial_params.copy()
        best_fitness = float('inf')
        
        for iteration in range(iterations):
            optimizer.zero_grad()
            
            input_vec = torch.cat([params, torch.tensor([alpha])])
            input_normalized = (input_vec - torch.tensor(surrogate_model.input_mean)) / torch.tensor(surrogate_model.input_std)
            
            output_normalized = surrogate_model.model(input_normalized.unsqueeze(0))
            output = output_normalized.squeeze(0) * torch.tensor(surrogate_model.output_std) + torch.tensor(surrogate_model.output_mean)
            
            cd = output[1]
            cl = output[0]
            
            fitness = cd / torch.clamp(cl, min=0.1)
            
            fitness.backward()
            optimizer.step()
            
            with torch.no_grad():
                params.data = torch.clamp(params.data, torch.tensor([-1.0]), torch.tensor([1.0]))
            
            current_fitness = fitness.item()
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_params = params.detach().numpy().copy()
            
            if iteration % 10 == 0:
                print(f"mwasifanwar Iteration {iteration}, Fitness: {current_fitness:.6f}")
        
        return best_params, best_fitness
    
    def compute_gradients(self, surrogate_model, params, alpha=5.0):
        params_tensor = torch.tensor(params, dtype=torch.float32, requires_grad=True)
        
        input_vec = torch.cat([params_tensor, torch.tensor([alpha])])
        input_normalized = (input_vec - torch.tensor(surrogate_model.input_mean)) / torch.tensor(surrogate_model.input_std)
        
        output_normalized = surrogate_model.model(input_normalized.unsqueeze(0))
        output = output_normalized.squeeze(0) * torch.tensor(surrogate_model.output_std) + torch.tensor(surrogate_model.output_mean)
        
        cd = output[1]
        cd.backward()
        
        return params_tensor.grad.numpy()