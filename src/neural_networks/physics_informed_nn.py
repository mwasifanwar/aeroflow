import torch
import torch.nn as nn
import numpy as np

class PhysicsInformedNN(nn.Module):
    def __init__(self, input_dim, hidden_layers=None):
        super(PhysicsInformedNN, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [64, 128, 64]
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            current_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        self.velocity_branch = nn.Sequential(
            nn.Linear(current_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )
        
        self.pressure_branch = nn.Sequential(
            nn.Linear(current_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        velocity = self.velocity_branch(shared_features)
        pressure = self.pressure_branch(shared_features)
        return torch.cat([velocity, pressure], dim=1)

class PINNSolver:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_dim = 3
        hidden_layers = self.config.get('neural_networks.physics_informed_layers')
        
        self.model = PhysicsInformedNN(input_dim, hidden_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get('neural_networks.learning_rate'))
        
        self.lambda_continuity = 1.0
        self.lambda_momentum = 1.0
        self.lambda_data = 1.0
    
    def physics_loss(self, outputs, inputs):
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        
        x = inputs[:, 0:1]
        y = inputs[:, 1:2]
        t = inputs[:, 2:3]
        
        u.requires_grad_(True)
        v.requires_grad_(True)
        p.requires_grad_(True)
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        continuity = u_x + v_y
        momentum_x = u_t + u * u_x + v * u_y + p_x
        momentum_y = v_t + u * v_x + v * v_y + p_y
        
        loss_continuity = torch.mean(continuity**2)
        loss_momentum = torch.mean(momentum_x**2 + momentum_y**2)
        
        return self.lambda_continuity * loss_continuity + self.lambda_momentum * loss_momentum
    
    def train(self, collocation_points, boundary_points, initial_points, epochs=1000):
        collocation_tensor = torch.FloatTensor(collocation_points).to(self.device).requires_grad_(True)
        boundary_tensor = torch.FloatTensor(boundary_points[:, :3]).to(self.device)
        boundary_values = torch.FloatTensor(boundary_points[:, 3:]).to(self.device)
        initial_tensor = torch.FloatTensor(initial_points[:, :3]).to(self.device)
        initial_values = torch.FloatTensor(initial_points[:, 3:]).to(self.device)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            collocation_outputs = self.model(collocation_tensor)
            physics_loss = self.physics_loss(collocation_outputs, collocation_tensor)
            
            boundary_outputs = self.model(boundary_tensor)
            boundary_loss = torch.mean((boundary_outputs - boundary_values)**2)
            
            initial_outputs = self.model(initial_tensor)
            initial_loss = torch.mean((initial_outputs - initial_values)**2)
            
            total_loss = physics_loss + self.lambda_data * (boundary_loss + initial_loss)
            
            total_loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                print(f"mwasifanwar Epoch {epoch}, Total Loss: {total_loss.item():.6f}")
    
    def predict(self, points):
        self.model.eval()
        with torch.no_grad():
            points_tensor = torch.FloatTensor(points).to(self.device)
            outputs = self.model(points_tensor)
            return outputs.cpu().numpy()