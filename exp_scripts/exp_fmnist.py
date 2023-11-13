from src_exp.exp_pipeline_classification import ExpPipelineClassification
from neural_nets.nets import FFN, CongruentFNN
from src_data.fmnist import Fmnist
from exp_scripts.exp_plot import ExpPlot

import numpy as np


class FmnistExp(object):
    def __init__(self):
        n_runs = 5
        n_agents = 10
        n_rounds = 10
        epoch_list = [10, 100, 1000]
        tasks = ['fmnist']
        agent_ids = [str(_) for _ in range(n_agents)]
        for exp_id in range(n_runs):
            fm = Fmnist(data_path='../exp_data/', save_path='../exp_data/FashionMNIST/data/')
            config = {'test_ratio': 0.94,
                      'data_path': fm.save_path,
                      'tasks': tasks,
                      'temp_path': '../temp/fmnist/',
                      'save_dir': f'../results/fmnist/n_agents_{n_agents}/',
                      'learning_rate': 1e-3,
                      'batch_size': 64}

            config_fl = {'num_epochs': None,
                         'n_rounds': n_rounds,
                         'agent_ids': agent_ids}

            config_cfl = {'n_rounds_fl': n_rounds,
                          'num_epochs_fl': None,
                          'agent_ids': agent_ids}

            seed = np.random.randint(0, 1e6, 1)
            net = FFN(in_dim=784, out_dim=10,
                      hidden_dim=[50]*2, activation=['relu']*2, seed=seed)
            prob_net = CongruentFNN(in_dim=784, out_dim=10,
                                    hidden_dim=[50]*2, activation=['relu']*2, seed=seed)

            print('*************************************')
            print('experiment id %s' % str(exp_id))

            exp = ExpPipelineClassification(config=config, exp_name=str(exp_id))
            for epoch in epoch_list:
                print(f'*** FL{epoch} ***')
                config_fl.update({'num_epochs': epoch})
                exp.federated_training(net, config_fl, save_name=f'fl{epoch}')

            for epoch in epoch_list:
                print(f'*** CFL{epoch} ***')
                config_cfl.update({'num_epochs_fl': epoch})
                exp.congruent_federated_training(prob_net, config_cfl, save_name=f'cfl{epoch}')

        ExpPlot(n_agents=n_agents, task_list=tasks, n_rounds=n_rounds, n_runs=n_runs, epoch_list=epoch_list)()


if __name__ == '__main__':
    FmnistExp()
