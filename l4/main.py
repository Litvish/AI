import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)
activation_function = nn.Tanh()

function_select = 4

def myfun(x):
    return np.exp(x/2)


batch_size = 100
x_train_fix = np.linspace(-5, 5, num=batch_size).reshape(-1,1)


x_eval_fix = np.linspace(-5, 5, num=batch_size).reshape(-1,1)


def train_model_simple(x_train, y_train, x_eval, units, epochs):
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    x_eval_tensor = torch.from_numpy(x_eval).float().to(device)

    y_train_tensor = torch.from_numpy(y_train).float().to(device)


    layer1 = nn.Linear(x_train.shape[1], units).to(device)
    layer2 = nn.Linear(units, 1, bias=False).to(device)


    parameters = list(layer1.parameters()) + list(layer2.parameters())


    optimizer = optim.Adam(parameters)
    loss_fn = nn.MSELoss(reduction='mean')


    for epoch in range(epochs):
        yhat = layer2(activation_function(layer1(x_train_tensor)))

        loss = loss_fn(yhat, y_train_tensor)


        loss.backward()


        optimizer.step()

        optimizer.zero_grad()

    yhat_eval = layer2(activation_function(layer1(x_eval_tensor)))

    return yhat.detach().cpu().numpy(), yhat_eval.detach().cpu().numpy()


def approx_1d_function(x_train, x_eval, units, epochs):
    y_train = myfun(x_train)
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaled = x_scaler.fit_transform(x_train_fix)
    y_scaled = y_scaler.fit_transform(y_train)
    x_eval_scaled = x_scaler.transform(x_eval_fix)

    _, result_eval = train_model_simple(x_scaled, y_scaled, x_eval_scaled, units, epochs)

    res_rescaled = y_scaler.inverse_transform(result_eval)

    y_eval = myfun(x_eval)

    return x_eval, res_rescaled, y_eval

def plot_1d_function(x_train, x_eval, predictions, labels, units, epochs):
    fig = plt.figure(1, figsize=(18,6))
    ax = fig.add_subplot(1, 2, 1)
    ax.axvspan(x_train.flatten()[0], x_train.flatten()[-1], alpha=0.15, color='green')
    plt.plot(x_eval, myfun(x_eval), '-', color='white', linewidth=1.0)
    plt.plot(x_eval, predictions, '-', label='output', color='yellow', linewidth=2.0)
    plt.plot(x_train, myfun(x_train), '.', color='black')
    plt.grid(which='both');
    plt.rcParams.update({'font.size': 12})
    plt.xlabel('x');
    plt.ylabel('y')
    plt.title('%d neurons in hidden layer with %d epochs of training' % (units ,epochs))
    plt.legend(['Function f(x)', 'MLP output g(x)', 'Training set'])
    ax = fig.add_subplot(1, 2, 2)
    ax.axvspan(x_train.flatten()[0], x_train.flatten()[-1], alpha=0.15, color='green')
    plt.plot(x_eval, np.abs(predictions-myfun(x_eval)), '-', label='output', color='firebrick', linewidth=2.0)
    plt.grid(which='both');
    plt.xlabel('x');
    plt.ylabel('y')
    plt.title('Absolute difference between prediction and actual function')
    plt.legend(['Error |f(x)-g(x)|'])
    plt.show()

units = 3
epochs = 10000
x, predictions, labels = approx_1d_function(x_train=x_train_fix, x_eval=x_eval_fix, units=units, epochs=epochs)
plot_1d_function(x_train_fix, x_eval_fix, predictions, labels, units, epochs)
