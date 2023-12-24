import cfl.src.congruent_mask as mask

import torch
from torch import nn


class FFN(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim: list, activation: list, seed=None):

        self.activation = [_.lower() for _ in activation]

        self.drop_out = 0.
        if seed is not None:
            torch.manual_seed(seed)
        super(FFN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self._update(seed=None, drop_out=self.drop_out)

    def update(self, seed, drop_out=None):
        if drop_out is None:
            drop_out = self.drop_out
        self._update(seed, drop_out)

    def _update(self, seed, drop_out):
        if seed is not None:
            torch.manual_seed(seed)

        module_dict = {'net_input': {'size': [self.in_dim, self.hidden_dim[0]],
                                     'activation': self.activation[0]}}
        for i in range(1, len(self.hidden_dim)):
            module_dict.update({'net_hidden' + str(i): {'size': [self.hidden_dim[i-1], self.hidden_dim[i]],
                                                        'activation': self.activation[i]}})
        module_dict.update({'net_output': {'size': [self.hidden_dim[-1], self.out_dim],
                                           'activation': ''}})

        for module_name in module_dict.keys():
            module_size = module_dict[module_name]['size']
            module_activation = module_dict[module_name]['activation']
            self.add_module(module_name, nn.Sequential(nn.Linear(in_features=module_size[0],
                                                                 out_features=module_size[1]))
                            )
            if module_activation != '':
                if module_activation.lower() == 'tanh':
                    self.add_module(module_name + '_activation', nn.Sequential(
                                                nn.Tanh(),
                                                nn.Dropout(drop_out),
                                                nn.BatchNorm1d(module_size[1], affine=False)
                    )
                                    )
                elif module_activation.lower() == 'selu':
                    self.add_module(module_name + '_activation', nn.Sequential(
                                                nn.SELU(True),
                                                nn.Dropout(drop_out),
                                                nn.BatchNorm1d(module_size[1], affine=False)
                    )
                                    )
                elif module_activation.lower() == 'relu':
                    self.add_module(module_name + '_activation', nn.Sequential(
                                                nn.ReLU(True),
                                                nn.Dropout(drop_out),
                                                nn.BatchNorm1d(module_size[1], affine=False)
                    )
                                    )
                else:
                    raise NotImplemented

    def forward(self, x):

        for module_name in list(self._modules.keys()):
            x = self._modules[module_name](x)
        return x


class CongruentFNN(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim: list, activation: list, seed=None):
        self.mask_names = ['mask_input']
        for i in range(1, len(hidden_dim)):
            self.mask_names.append('mask_hidden' + str(i))
        self.mask_names.append('mask_output')
        self.mask_names.append('mask_input_std')
        for i in range(1, len(hidden_dim)):
            self.mask_names.append('mask_hidden' + str(i) + '_std')
        self.mask_names.append('mask_output_std')
        self.activation = [_.lower() for _ in activation]

        self.drop_out = 0.
        if seed is not None:
            torch.manual_seed(seed)
        super(CongruentFNN, self).__init__()
        self.mask_dict = {}
        for mask_name in self.mask_names:
            self.mask_dict.update({mask_name: None})
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self._update(seed=None, drop_out=self.drop_out)

    def update(self, seed, drop_out=None):
        if drop_out is None:
            drop_out = self.drop_out
        self._update(seed, drop_out)

    def _update(self, seed, drop_out):
        if seed is not None:
            torch.manual_seed(seed)

        module_dict = {'net_input': {'size': [self.in_dim, self.hidden_dim[0]],
                                     'activation': self.activation[0]}}
        for i in range(1, len(self.hidden_dim)):
            module_dict.update({'net_hidden' + str(i): {'size': [self.hidden_dim[i-1], self.hidden_dim[i]],
                                                        'activation': self.activation[i]}})
        module_dict.update({'net_output': {'size': [self.hidden_dim[-1], self.out_dim],
                                           'activation': ''}})

        for module_name in module_dict.keys():
            module_size = module_dict[module_name]['size']
            module_activation = module_dict[module_name]['activation']
            self.add_module(module_name, nn.Sequential(
                mask.ProbCongruentMask(in_features=module_size[0], out_features=module_size[1]))
                            )
            if module_activation != '':
                if module_activation.lower() == 'tanh':
                    self.add_module(module_name + '_activation', nn.Sequential(
                                                nn.Tanh(),
                                                nn.Dropout(drop_out),
                                                nn.BatchNorm1d(module_size[1], affine=False)
                    )
                                    )
                elif module_activation.lower() == 'selu':
                    self.add_module(module_name + '_activation', nn.Sequential(
                                                nn.SELU(True),
                                                nn.Dropout(drop_out),
                                                nn.BatchNorm1d(module_size[1], affine=False)
                    )
                                    )
                elif module_activation.lower() == 'relu':
                    self.add_module(module_name + '_activation', nn.Sequential(
                                                nn.ReLU(True),
                                                nn.Dropout(drop_out),
                                                nn.BatchNorm1d(module_size[1], affine=False)
                    )
                                    )
                else:
                    raise NotImplemented

    def forward(self, x):
        mask_names = list(self.mask_dict.keys())
        std_mask = list(filter(lambda m: 'std' in m, mask_names))
        for _ in std_mask:
            mask_names.remove(_)
        for mask_name in mask_names:
            net = self._modules['net' + mask_name[4:]]
            x = net((x, self.mask_dict[mask_name], self.mask_dict[mask_name + '_std']))
            if mask_name != 'mask_output':
                net_activation = self._modules['net' + mask_name[4:] + '_activation']
                x = net_activation(x)
        return x
