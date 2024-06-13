########################################################################################################################
## step 1: loading the required packages
import numpy as np
import pylab as pl
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
from datetime import datetime, timedelta

print("Loading packages ---> done!")
########################################################################################################################
########################################################################################################################
########################################################################################################################
save_folder_name = "results_DE"
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, save_folder_name)):
    os.mkdir(os.path.join(cwd, save_folder_name))

print("Check directory ---> done!")
########################################################################################################################
########################################################################################################################
########################################################################################################################
## step 2: setting the plotting properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 22}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (12, 8)


########################################################################################################################
########################################################################################################################
########################################################################################################################
## step 2: define functions

def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = np.reshape(sequences[i:end_ix, :], n_steps_in * sequences.shape[-1]), \
            sequences[end_ix:out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def to_np(x):
    return x.cpu().detach().numpy()


class FCN_model(nn.Module):
    "Defines a connected network"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, epsilon=1e-8):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])

        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])

        self.fce = nn.Linear(N_HIDDEN, 2 * N_OUTPUT)
        self.N_OUTPUT = N_OUTPUT
        self.epsilon = epsilon
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        mu = x[:, :self.N_OUTPUT]
        sigma = self.softplus(x[:, self.N_OUTPUT:]) + self.epsilon

        return mu, sigma


########################################################################################################################
########################################################################################################################
########################################################################################################################
## step 3: parameters of training
learning_rate = 0.0005
epsilon = 1e-8

num_iter = 500
batch_size = 256

num_networks = 10
num_print = 10
CTE = 10

default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
########################################################################################################################
########################################################################################################################
########################################################################################################################

## step 4: read data
EAM = np.loadtxt("./Manuscript_all/data/EAM.txt")

all_dates = [datetime.strptime("2021-09-01", "%Y-%m-%d") + i * timedelta(days=7) for i in range(70)]

for epoch_number in range(len(all_dates)):
    name = f"./Manuscript_all/data/{str(all_dates[epoch_number])[:10]}-IERS.txt"
    dUT1 = np.loadtxt(name)
    dUT1 = dUT1.reshape(dUT1.shape[0], 1)
    tmp_EAM = EAM[:dUT1.shape[0], :]

    data = np.concatenate((dUT1, tmp_EAM), axis=1)

    X_train, Y_train = split_sequences(data, 30, 10)
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    ## step 5: prepare the data
    X_train = Tensor(X_train)
    Y_train = Tensor(Y_train)
    X_test = Tensor(np.reshape(data[-30:, :], (1, X_train.shape[1]))).to(default_device)

    train_dataset = TensorDataset(X_train.to(default_device), Y_train.to(default_device))
    trainloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ## step 6: defining the models
    model_list = []
    opt_list = []

    for i in range(num_networks):
        model_list.append(FCN_model(X_train.shape[-1], Y_train.shape[-1], 10, 3, epsilon).to(default_device))
        opt_list.append(torch.optim.Adam(model_list[i].parameters(), lr=learning_rate))

    train_data_num = X_train.shape[0]
    test_data_num = X_test.shape[0]

    loss_train = np.zeros([num_networks])
    out_mu = []
    out_sig = []

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ## step 7: training the models

    for iter in range(num_iter):
        for i in range(num_networks):
            model_list[i].train()
            tmp_loss = 0.
            for x_batch, y_batch in trainloader:
                opt_list[i].zero_grad()
                mu_train, sig_train = model_list[i](x_batch)

                loss = torch.mean(
                    torch.log(sig_train) + torch.divide(torch.abs(y_batch - mu_train), sig_train)) + CTE
                tmp_loss += loss
                if np.any(np.isnan(to_np(loss))):
                    print(torch.divide(torch.abs(y_batch - mu_train), sig_train))
                    raise ValueError('There is Nan in loss')

                loss.backward()
                opt_list[i].step()

            loss_train[i] += tmp_loss.item()

        if iter % num_print == 0 and iter != 0:
            print(('-------------------------') + ' Iteration: ' + str(iter) + ' -------------------------')
            print('Average Loss(NLL): ' + str(loss_train / num_print))
            print('\n')

            loss_train = np.zeros(num_networks)

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    # step 8: predict for the test set

    for i in range(num_networks):
        model_list[i].eval()
        mu_test, sig_test = model_list[i](X_test)

        out_mu.append(to_np(mu_test))
        out_sig.append(to_np(sig_test))

    out_mu = np.array(out_mu)
    out_sig = np.array(out_sig)

    out_mu_final = np.mean(out_mu, axis=0)
    out_sig_final = np.sqrt(
        np.mean(2 * np.square(out_sig), axis=0) + np.mean(np.square(out_mu), axis=0) - np.square(out_mu_final))

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    ## step 9: save the prediction results

    np.savetxt(os.path.join(cwd, save_folder_name, f"mean_LDE_{epoch_number+1}.txt"), out_mu_final)
    np.savetxt(os.path.join(cwd, save_folder_name, f"std_LDE_{epoch_number+1}.txt"), out_sig_final)
