import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[4],[7],[9]], dtype = torch.float32)
Y = torch.tensor([[2],[4],[8],[14],[18]], dtype = torch.float32)

X_test = torch.tensor([[5]], dtype=torch.float32)

n_sample, n_features = X.shape
model = nn.Linear(n_features, n_features)

learn_rate = 0.01
n_epochs = 500

loss = nn.MSELoss()
optimizer = torch.optim.SGD(lr = learn_rate, params=model.parameters())

for i in range(0,n_epochs):
    y_pred = model(X)
    ls = loss(y_pred,Y)
    ls.backward()
    optimizer.step()
    optimizer.zero_grad()
    [w,b] = model.parameters()
    traced_cell = torch.jit.trace(model, X_test)
    #print(traced_cell(X_test).item())
    print(f"{ls.item():0.3f}, {w[0][0].item():0.3f}, {model(X_test).item():0.3f}")

traced_cell.save('model1.zip')