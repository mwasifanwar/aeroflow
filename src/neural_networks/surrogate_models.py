import torch
import torch.nn as nn
import numpy as np

class SurrogateModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None):
        super(SurrogateModel, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 256, 128]
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class AerodynamicSurrogate:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_dim = self.config.get('geometry.parameter_dim') + 1
        output_dim = 3
        hidden_layers = self.config.get('neural_networks.surrogate_hidden_layers')
        
        self.model = SurrogateModel(input_dim, output_dim, hidden_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get('neural_networks.learning_rate'))
        self.criterion = nn.MSELoss()
        
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
    
    def train(self, design_parameters, angles, aerodynamic_coefficients, epochs=None):
        if epochs is None:
            epochs = self.config.get('neural_networks.epochs')
        
        inputs = []
        for params, alpha in zip(design_parameters, angles):
            input_vec = np.concatenate([params, [alpha]])
            inputs.append(input_vec)
        
        inputs = np.array(inputs)
        outputs = np.array(aerodynamic_coefficients)
        
        inputs_tensor, self.input_mean, self.input_std = self._normalize_tensor(inputs)
        outputs_tensor, self.output_mean, self.output_std = self._normalize_tensor(outputs)
        
        dataset = torch.utils.data.TensorDataset(inputs_tensor, outputs_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.get('neural_networks.batch_size'), shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_inputs, batch_outputs in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_inputs)
                loss = self.criterion(predictions, batch_outputs)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 100 == 0:
                print(f"mwasifanwar Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
    
    def predict(self, design_parameters, angle):
        self.model.eval()
        with torch.no_grad():
            input_vec = np.concatenate([design_parameters, [angle]])
            input_normalized = (input_vec - self.input_mean) / self.input_std
            input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0).to(self.device)
            output_normalized = self.model(input_tensor)
            output = output_normalized.cpu().numpy()[0] * self.output_std + self.output_mean
        
        return {'cl': output[0], 'cd': output[1], 'cm': output[2]}
    
    def _normalize_tensor(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1.0, std)
        normalized = (data - mean) / std
        return torch.FloatTensor(normalized).to(self.device), mean, std
    
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.input_mean = checkpoint['input_mean']
        self.input_std = checkpoint['input_std']
        self.output_mean = checkpoint['output_mean']
        self.output_std = checkpoint['output_std']