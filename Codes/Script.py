import numpy as np
import torch.nn as nn
import torchvision
import torch.utils.data as Data 
import torch
from sklearn import tree, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Question 1
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3, random_state=42)

#Arbre de décision
clf = tree.DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

#Prédiction sur la base test
y_pred_tree = clf.predict(X_test)

#La fonction perte F
tree_loss = np.mean(np.abs(y_pred_tree - y_test))
print("Fonction perte (Arbre de décision) base test :", tree_loss)

#Question 2
'''
Pour bien choisir la base d’entraînement dans notre cas, il est important de porter notre attention sur le partitionnement de notre base.
Nous avons 150 exemples répartis en 3 classes (50 + 50 + 50), de ce fait nous pouvons choisir une base d’entraînement plus grande (2/3 des 150 exemples)
pour que notre modèle apprenne les différents motifs dans la base de données et une base test plus petite.
Cette répartition permet d’entraîner notre modèle avec un bon nombre d’exemples sans prendre toute la base de données ce qui entraînera un surapprentissage.
De plus les exemples dans chaque partition de notre base de données doivent être tirés de manière aléatoire parmi les 150 exemples
pour éviter les biais causés par l’ordre des données, d’où le paramètre random_state dans notre fonction train_test_split.

En ce qui concerne la fonction perte dans notre contexte, l'arbre de décision peut classer les exemples de la base test de manière à faire peu d'erreur à cause de la simplicité. De ce fait si l'arbre de décision
parvient à classer parfaitement les 50 exemples de la base test nous aurons Y_n = y_n donc F = 0, sinon, si quelques exemples sont mal classifiés la fonction perte sera supérieur à 0 mais très faible.
'''

#Question 3
class ReseauNeurone(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReseauNeurone, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        logits = self.linear(x)
        out = self.softmax(logits)
        return out

input_dim = 4
output_dim = 3
model = ReseauNeurone(input_dim, output_dim)
#Fonction perte
loss_fn = nn.L1Loss()
#Gradient stochastique (SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

X_train_tensor = torch.FloatTensor(X_train)
#Pour que le format des étiquettes corresponde à celui de la sortie du réseau de neurones :
y_train_onehot = np.eye(output_dim)[y_train]
y_train_tensor = torch.FloatTensor(y_train_onehot)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)        
    loss = loss_fn(outputs, y_train_tensor)  
    loss.backward()                         
    optimizer.step()                        

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss.item()}")

#Question 4
model.eval()
with torch.no_grad(): 
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_onehot = np.eye(output_dim)[y_test]
    y_test_tensor = torch.FloatTensor(y_test_onehot)
    
    outputs_test = model(X_test_tensor)
    test_loss = loss_fn(outputs_test, y_test_tensor)

nn_loss = test_loss.item()

print(f"Arbre de decision : perte moyenne = {tree_loss:.4f}")
print(f"Reseau de neurones : perte moyenne = {nn_loss:.4f}")

'''
Nous remarquons que la parte moyenne de l'arbre de decision est minime, 0.0200 comparé à la perte moyenne de notre reseau de neurones simple. Cette difference est
du à la simplicite de notre reseau qui ne prend pas en compte, assez, des differentes subtilites dans les donnees alors que l'arbre de decision s'adapte aux differentes
subtilites.
'''

#Question 5
'''
Pour faciliter la lecture et la correction, je me suis permis d'écrire le code de la question 5 à part. Vous trouverez ce code ici : 
'''
