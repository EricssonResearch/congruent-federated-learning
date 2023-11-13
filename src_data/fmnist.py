import torchvision
import numpy as np
import os
import pickle


class Fmnist(object):
    def __init__(self, data_path: str, save_path: str, n_agents: int = 100, max_num_samples=500):
        data_path = data_path
        n_agents = n_agents

        data = torchvision.datasets.FashionMNIST(root=os.path.join(data_path, "FashionMNIST"),
                                                 train=True, download=True)
        x, y = data.data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data.targets)

        x = x.reshape(x.shape[0], x.shape[-1] * x.shape[-2])
        random_ids = np.random.randint(0, n_agents, x.shape[0])
        assert len(np.unique(random_ids)) == n_agents

        save_path += str(n_agents) + '_n' + str(max_num_samples).lower() + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for agent in range(n_agents):
            agent_ids = np.where(random_ids == agent)[0]
            if max_num_samples is not None:
                assert max_num_samples <= len(agent_ids)
                x_agent = x[agent_ids, :][0:max_num_samples, :]
                y_agent = y[agent_ids][0:max_num_samples]
            else:
                x_agent = x[agent_ids, :]
                y_agent = y[agent_ids]
            agent_dict = {'dataset': {'X': x_agent,
                                      'Y': y_agent},
                          'id': str(agent)}
            with open(save_path + str(agent) + '.pickle', 'wb') as handle:
                pickle.dump(agent_dict, handle)
        self.save_path = save_path


if __name__ == '__main__':
    DATA_PATH = '/Users/etagjal/Air/data/cfl_boosting/'
    Fmnist(data_path=DATA_PATH, save_path=DATA_PATH + 'FashionMNIST/data/', n_agents=10, max_num_samples=5000)
