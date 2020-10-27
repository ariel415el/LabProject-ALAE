import torch
from torch.autograd import grad

# This script served me as a simple experiment with torch.autograd.grad function that is used in ALAE for the R1 gradient
# regularization.
# It also includes a reminder for how the backpropagation algorithm works

# Create some dummy data.
x = torch.ones((1,), requires_grad=True)
gt = torch.ones_like(x) * 16 - 0.5  # "ground-truths"

## Forward:
v = x + 2  # = 3
y = v ** 2  # = 9
diff = (y - gt)  # = -6.5
loss = diff**2  # = 42.25

print(f"x: {x}")
print(f"v: {v}")
print(f"y: {y}")
print(f"diff: {diff}")
print(f"loss: {loss}")

## Backward:
# dloss/ddiff = 2diff = -13
# dloss/dy = dloss/ddiff * ddiff/dy = -13 * 1
# dloss/dv = dloss/y * dy/dv = -13 * (2*v) = -78
# dloss/dx = dloss/v * dv/dx = -78 * 1 = -78

print(f"dloss/loss: {grad(outputs=loss, inputs=loss, grad_outputs=torch.ones_like(loss))}")
print(f"dloss/ddiff: {grad(outputs=loss, inputs=diff, grad_outputs=torch.ones_like(loss), retain_graph=True)}")
print(f"dloss/dy: {grad(outputs=loss, inputs=y, grad_outputs=torch.ones_like(loss), retain_graph=True)}")
print(f"dloss/dv: {grad(outputs=loss, inputs=v, grad_outputs=torch.ones_like(loss), retain_graph=True)}")
print(f"dloss/dx: {grad(outputs=loss, inputs=x, grad_outputs=torch.ones_like(loss), retain_graph=True)}")

