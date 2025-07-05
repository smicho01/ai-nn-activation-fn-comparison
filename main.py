import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

x_train = torch.linspace(-10, 10, 1000).unsqueeze(1) # gen data between -10 and 10
y_train = x_train ** 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Used device =", device.type)

x_train = x_train.to(device)
y_train = y_train.to(device)


class NetWithoutActivation(nn.Module):
    def __init__(self):
        super(NetWithoutActivation, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)
    

class NetWithReLUActivation(nn.Module):
    def __init__(self):
        super(NetWithReLUActivation, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.model(x)
    

def train_model(model, x_train, y_train, epochs=1000, learning_rate=0.001):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return model


def plot_result(model, title):
    model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(-10, 10, 1000).unsqueeze(1).to(device)
        y_true = x_plot ** 3
        y_pred = model(x_plot)
    
    x_plot = x_plot.cpu().numpy()
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    plt.plot(x_plot, y_true, label='True $x^3$', color='blue')
    plt.plot(x_plot, y_pred, label='Network Predicted', color='red')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


print("Training without activation function")
model_no_activation = NetWithoutActivation()
model_no_activation = train_model(model_no_activation, x_train, y_train, epochs=1000, learning_rate=0.001)
plot_result(model_no_activation, "NN Without Activation Function")

print("Training with activation function (ReLU)")
model_with_activation = NetWithReLUActivation()
model_with_activation = train_model(model_with_activation, x_train, y_train, epochs=1000, learning_rate=0.001)
plot_result(model_with_activation, "NN With ReLU Activation Function")  