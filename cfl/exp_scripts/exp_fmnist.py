from cfl.src_exp.exp_pipeline_classification import ExpPipelineClassification
from cfl.neural_nets.nets import FFN, CongruentFNN
from cfl.exp_scripts.exp_plot import ExpPlot
from cfl.data import fg_traces

import numpy as np
from dataclasses import dataclass
from torch import nn
import pickle
from unittest import TestCase


@dataclass
class LearningConfig:
    """Contains all the configs realted to learning."""
    n_rounds: int = 10  # number of FL rounds
    n_epochs: int = 100  # number of epochs per round (CFL prefers to be large say 100)
    learning_rate: float = 1e-3  # learning rate of the Adam optimizer
    batch_size: int = 128  # learning batch size


@dataclass
class ExpConfig:
    """Contains all the configs of the experiments."""
    learning_config = LearningConfig  # define as above
    data_path = fg_traces.__path__[0] + '/'
    agent_ids = [str(_) for _ in range(10)]  # a list of strings corresponding to the name of agents
    test_ratio: float = 0.95  # split data into train and test set
    batch_size: int = 128  # learning batch size
    n_runs: int = 1  # number of runs of the experiment
    save_dir: str =  './fmnist_exp/'  #  directory where the results are stored
    net: nn.Module = FFN(in_dim=784,  # input dimension adjust according to your data
                         out_dim=10,  # output dimension
                         hidden_dim=[50, 50],  # number of units at each layer
                         activation=['relu', 'relu'],  # activation function at each hidden layer
                         seed=np.random.randint(0, 1e6, 1))  # this is an example torch model for vanilla FL
    congrunet_net: nn.Module = CongruentFNN(in_dim=784,  # input dimension
                                            out_dim=10,  # output dimension
                                            hidden_dim=[50, 50],  # number of units at each hidden layer
                                            activation=['relu', 'relu'],  # activation function at each layer
                                            seed=np.random.randint(0, 1e6, 1))  # this is an example torch model for  CFL
    make_predictions_using_local_model: bool = False  #  True if final predictions should be made at the client side

            
class _DataParser(object):
    """A simple verification of data structure. """
    def __init__(self, data_path: str, agent_ids: list):
        for agent in agent_ids:
            with open(data_path + f'{agent}.pickle', 'rb') as handle:
                agent_dict = pickle.load(handle)
            TestCase().assertListEqual(list(agent_dict.keys()), ['dataset', 'id'], 
                                       msg='Check the data structure!')
            TestCase().assertListEqual(list(agent_dict['dataset'].keys()), ['X', 'Y'], 
                                       msg='Check the structure of dataset!')
        print("agentData has correct structure!")


class ExpFMNIST(object):
    """Experiment using both FL and CFL."""
    def __init__(self, exp_config: ExpConfig):
        _DataParser(data_path=exp_config.data_path, agent_ids=exp_config.agent_ids)
        config = {'test_ratio': exp_config.test_ratio,
                  'data_path': exp_config.data_path,
                  'tasks':  ['FMNIST'],
                  'learning_rate': exp_config.learning_config.learning_rate,
                  'batch_size': exp_config.learning_config.batch_size,
                  'temp_path': exp_config.save_dir + 'temp/',
                  'save_dir': exp_config.save_dir,
                  'use_best_model': False,  
                  'val_ratio': None
                  }
        

        config_fl = {'num_epochs': exp_config.learning_config.n_rounds,
                     'n_rounds':  exp_config.learning_config.n_epochs,
                     'agent_ids': exp_config.agent_ids}

        config_cfl = {'n_rounds_fl': exp_config.learning_config.n_rounds,
                      'num_epochs_fl': exp_config.learning_config.n_epochs,
                      'agent_ids': exp_config.agent_ids}


        for exp_id in range(exp_config.n_runs):
            print('*************************************')
            print('experiment id %s' % str(exp_id))

            exp = ExpPipelineClassification(config=config, exp_name=str(exp_id))

            print('*** FL ***')
            exp.federated_training(exp_config.net, config_fl, save_name=f'fl{exp_config.learning_config.n_epochs}')

            print('*** CFL ***')
            exp.congruent_federated_training(exp_config.congrunet_net, config_cfl, save_name=f'cfl{exp_config.learning_config.n_epochs}')
        
        ExpPlot(save_path=exp_config.save_dir, 
                n_agents=len(exp_config.agent_ids), 
                task_list=config['tasks'], 
                n_rounds=exp_config.learning_config.n_rounds, 
                n_runs=exp_config.n_runs, 
                epoch_list=[exp_config.learning_config.n_epochs,],
                make_predictions_using_local_model=exp_config.make_predictions_using_local_model)()


if __name__ == '__main__':
    e5g = ExpFMNIST(exp_config=ExpConfig())