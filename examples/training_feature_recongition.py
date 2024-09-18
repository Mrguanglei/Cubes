import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flexicubes1 import FeatureRecognitionNet
# Load your dataset
# X_train should be the training data and y_train should be the labels
# For demonstration purposes, let's assume they are tensors

X_train = torch.randn(100, 1, 32, 32)  # Example data
y_train = torch.randint(0, 2, (100, 1)).float()  # Example labels

dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = FeatureRecognitionNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # Number of epochs
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'models/feature_recognition_net.pth')
