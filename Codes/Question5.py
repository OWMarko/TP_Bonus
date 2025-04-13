import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import tree, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Chargement et préparation des données Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3, random_state=42)

# Conversion en tenseurs
input_dim = 4
output_dim = 3
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_onehot = np.eye(output_dim)[y_train]
y_test_onehot = np.eye(output_dim)[y_test]
y_train_tensor = torch.FloatTensor(y_train_onehot)
y_test_tensor = torch.FloatTensor(y_test_onehot)

#Nos réseaux
class ReseauNeurone_1_couches(nn.Module):
    def __init__(self):
        super(ReseauNeurone_1_couches, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 6),
            nn.Tanh(),
            nn.Linear(6, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

class ReseauNeurone_2_couches(nn.Module):
    def __init__(self):
        super(ReseauNeurone_2_couches, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 10),
            nn.Tanh(),
            nn.Linear(10, 6),
            nn.Tanh(),
            nn.Linear(6, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

class ReseauNeurone_3_couches(nn.Module):
    def __init__(self):
        super(ReseauNeurone_3_couches, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 12),
            nn.Tanh(),
            nn.Linear(12, 10),
            nn.Tanh(),
            nn.Linear(10, 8),
            nn.Tanh(),
            nn.Linear(8, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

class ReseauNeurone_4_couches(nn.Module):
    def __init__(self):
        super(ReseauNeurone_4_couches, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 12),
            nn.Tanh(),
            nn.Linear(12, 8),
            nn.Tanh(),
            nn.Linear(8, 6),
            nn.Tanh(),
            nn.Linear(6, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

class ReseauNeurone_10_couches(nn.Module):
    def __init__(self):
        super(ReseauNeurone_10_couches, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.Sigmoid(),
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16),
            nn.Sigmoid(),
            nn.Linear(16, 12),
            nn.Sigmoid(),
            nn.Linear(12, 12),
            nn.Sigmoid(),
            nn.Linear(12, 8),
            nn.Sigmoid(),
            nn.Linear(8, 6),
            nn.Sigmoid(),
            nn.Linear(6, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

#Entraînement
def train_and_evaluate(model, epochs=500, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    test_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
      
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                outputs_test = model(X_test_tensor)
                test_loss = loss_fn(outputs_test, y_test_tensor).item()
                test_losses.append(test_loss)
            print(f"Epoch {epoch}: Test loss = {test_loss:.4f}")
    return test_losses

# Initialisation des modèles
models = {
    "1 couche cachée": ReseauNeurone_1_couches(),
    "2 couches cachées": ReseauNeurone_2_couches(),
    "3 couches cachées": ReseauNeurone_3_couches(),
    "4 couches cachées": ReseauNeurone_4_couches(),
    "10 couches cachées (Sigmoid)": ReseauNeurone_10_couches(),
}

# Entraînement et collecte des pertes
results = {}
for name, model in models.items():
    print(f"\n--- Entraînement du modèle avec {name} ---")
    losses = train_and_evaluate(model)
    results[name] = losses

# Tracé des pertes
plt.figure(figsize=(10, 6))
for name, losses in results.items():
    plt.plot(range(0, 500, 100), losses, label=name)
plt.xlabel("Nombre d'époques")
plt.ylabel("Perte moyenne sur la base test")
plt.title("Comparaison des pertes pour différents réseaux")
plt.legend()
plt.grid(True)
plt.show()

