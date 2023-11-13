import pandas as pd


class SimpleRead(object):
    name = 'simple read'

    def __init__(self, data_path, tasks):
        self.data_path = data_path
        self.tasks = tasks

    def read(self, agent_id):
        return pd.read_pickle(self.data_path + agent_id + '.pickle')


class EstimatorRead(object):
    name = 'estimator read'

    def __init__(self, data_path, tasks, agent_id):
        self.data_path = data_path
        self.tasks = tasks
        self.agent_id = agent_id

    def read(self, agent_id):
        if agent_id:
            pass
        return pd.read_pickle(self.data_path + self.agent_id + '.pickle')
