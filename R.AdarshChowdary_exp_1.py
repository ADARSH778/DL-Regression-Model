import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

x = torch.linspace(1, 50, 50).reshape(-1, 1)

torch.manual_seed(71)

e = torch.randint(-8, 9, (50, 1)).float()

print(e.sum())

"""Reddineni Adarsh Chowdary
212223040166
"""

y = 2 * x + 1 + e
print(y.shape)

plt.scatter(x.numpy(), y.numpy(), color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()

torch.manual_seed(59)

model = nn.Linear(1, 1)

print('Weight:', model.weight.data.item())
print('Bias:  ', model.bias.data.item())

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 50
losses = []

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_function(y_pred, y)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()

    print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.weight.data.item():10.8f}  '
          f'bias: {model.bias.data.item():10.8f}')

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()

x1 = torch.tensor([x.min().item(), x.max().item()])
w1, b1 = model.weight.data.item(), model.bias.data.item()
y1 = x1 * w1 + b1

print(f'Final Weight: {w1:.8f}, Final Bias: {b1:.8f}')
print(f'X range: {x1.numpy()}')
print(f'Predicted Y values: {y1.numpy()}')

plt.scatter(x.numpy(), y.numpy(), label="Original Data")
plt.plot(x1.numpy(), y1.numpy(), 'r', label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'linear_regression_model.pth')
print('Model saved successfully.')
