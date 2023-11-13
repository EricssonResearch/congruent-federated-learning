from tools.helpers import TorchFunctions
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import os


class MLPClassifier(object):
    name = 'mlp classifier'

    def __init__(self, model, num_epochs, learning_rate, batch_size, use_best_model, min_num_epochs, save_path,
                 if_print=False, print_freq=.1):
        self.use_best_model = use_best_model
        self.min_num_epochs = min_num_epochs
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.if_print = if_print
        self.print_freq = print_freq
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self._fit_flag = False
        self._best_model = model
        self._best_loss = np.inf

    @property
    def best_loss(self):
        return self._best_loss

    @property
    def fit_flag(self):
        return self._fit_flag

    @property
    def best_model(self):
        return self._best_model

    def set(self, attribute, value):
        self.__setattr__(attribute, value)

    def fit(self, x_train, y_train, x_val, y_val):
        func = TorchFunctions()
        self.batch_size = min(self.batch_size, x_train.shape[0])

        x_train_loader = DataLoader(x_train, batch_size=self.batch_size, shuffle=False, drop_last=True)
        y_train_loader = DataLoader(y_train, batch_size=self.batch_size, shuffle=False, drop_last=True)

        val_loss = []
        for epoch in range(self.num_epochs):
            for (x_batch, y_batch) in zip(x_train_loader, y_train_loader):
                output = self.model.forward(func.make_float_variable(x_batch))

                train_loss = self.criterion(output, func.make_long_variable(y_batch).squeeze())

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
            val_loss.append(self._predict(x_val, y_val, eval_mode=True, as_numpy=True)[0])
            if (self.if_print is True) and (epoch % (self.num_epochs * self.print_freq) == 0):
                print('Train:      epoch [{}/{}], loss:{:.4f}'.format(epoch + 1,
                                                                      self.num_epochs, train_loss.item()))
                print('Validation: epoch [{}/{}], loss:{:.4f}'.format(epoch + 1,
                                                                      self.num_epochs, val_loss[-1]))
            if self.use_best_model is True and (epoch > self.min_num_epochs):
                if val_loss[-1] < self._best_loss:
                    torch.save(self.model, self.save_path + 'model.pickle')
                    self._best_loss = val_loss[-1].copy()
        if self.use_best_model is True and (self.num_epochs > self.min_num_epochs):
            self.model = torch.load(self.save_path + 'model.pickle')
        self._best_model = self.model
        self._fit_flag = True

    def _predict(self, x, y, eval_mode=True, as_numpy=True):
        func = TorchFunctions()
        n_samples = x.shape[0]
        x_loader = DataLoader(x, batch_size=n_samples, shuffle=False, drop_last=False)
        y_loader = DataLoader(y, batch_size=n_samples, shuffle=False, drop_last=False)
        x_data = func.make_float_variable(next(iter(x_loader)))
        y_data = func.make_long_variable(next(iter(y_loader))).squeeze()
        if eval_mode is True:
            self.model.eval()

        y_hat = self.model.forward(x_data)
        loss = self.criterion(y_hat, y_data)
        if as_numpy is False:
            return loss, torch.log_softmax(y_hat, dim=1)
        else:
            return loss.detach().numpy(), torch.log_softmax(y_hat, dim=1).detach().numpy()

    def predict(self, x):
        func = TorchFunctions()
        n_samples = x.shape[0]
        x_loader = DataLoader(x, batch_size=n_samples, shuffle=False, drop_last=False)
        x_data = func.make_float_variable(next(iter(x_loader)))
        self._best_model.eval()
        y_hat = self._best_model.forward(x_data)
        return torch.log_softmax(y_hat, dim=1).detach().numpy()

