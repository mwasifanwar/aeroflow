import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFlowPredictor(nn.Module):
    def __init__(self, input_channels=2, output_channels=3):
        super(CNNFlowPredictor, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, output_channels, 3, padding=1)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class GeometryToFlowCNN:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = CNNFlowPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get('neural_networks.learning_rate'))
        self.criterion = nn.MSELoss()
    
    def train(self, geometry_maps, flow_fields, epochs=500):
        geometry_tensor = torch.FloatTensor(geometry_maps).to(self.device)
        flow_tensor = torch.FloatTensor(flow_fields).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(geometry_tensor, flow_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_geom, batch_flow in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_geom)
                loss = self.criterion(predictions, batch_flow)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 50 == 0:
                print(f"mwasifanwar Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
    
    def predict(self, geometry_map):
        self.model.eval()
        with torch.no_grad():
            geometry_tensor = torch.FloatTensor(geometry_map).unsqueeze(0).to(self.device)
            prediction = self.model(geometry_tensor)
            return prediction.squeeze(0).cpu().numpy()
    
    def create_geometry_map(self, mesh_points, airfoil_points, resolution=(128, 128)):
        geometry_map = np.zeros(resolution)
        
        x_min, x_max = -2, 3
        y_min, y_max = -2, 2
        
        for i in range(resolution[0]):
            for j in range(resolution[1]):
                x = x_min + (x_max - x_min) * i / (resolution[0] - 1)
                y = y_min + (y_max - y_min) * j / (resolution[1] - 1)
                
                distances = np.linalg.norm(airfoil_points - [x, y], axis=1)
                if np.min(distances) < 0.05:
                    geometry_map[i, j] = 1.0
        
        return geometry_map