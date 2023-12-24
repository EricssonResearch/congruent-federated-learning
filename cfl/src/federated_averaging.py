import torch
from copy import deepcopy

EPSILON = 1e-16


class IsolatedTraining(object):
    name = 'isolated training'

    @staticmethod
    def learn_local(x_train, y_train, x_val, y_val, trainer):
        trainer.fit(x_train, y_train, x_val, y_val)
        return trainer

    @staticmethod
    def predict_local(x, trainer):
        return trainer.predict(x)

    @staticmethod
    def evaluate(y, y_hat, evaluator):
        return evaluator.evaluate(y, y_hat)


class FederatedAveraging(IsolatedTraining):
    name = 'federated averaging'

    def __init__(self, convergence_evaluator):
        self.convergence_evaluator = convergence_evaluator

    @staticmethod
    def average(agent_model_dict):
        agent_names = list(agent_model_dict.keys())
        n_agents = len(agent_names)
        layers = list(agent_model_dict[agent_names[0]].best_model.state_dict().keys())

        average_model = deepcopy(agent_model_dict[agent_names[0]])
        state_dict_ave = deepcopy(agent_model_dict[agent_names[0]].best_model.state_dict())
        for layer in layers:
            state_dict_layer_ave = agent_model_dict[agent_names[0]].best_model.state_dict()[layer] / n_agents
            for agent in agent_names[1:]:
                state_dict_layer_ave += (agent_model_dict[agent].best_model.state_dict()[layer] / n_agents)
            state_dict_ave.update({layer: state_dict_layer_ave})

        average_model.best_model.state_dict().update(state_dict_ave)
        average_model.best_model.load_state_dict(state_dict_ave)
        average_model.model.state_dict().update(state_dict_ave)
        average_model.model.load_state_dict(state_dict_ave)
        return average_model

    @staticmethod
    def std(agent_model_dict, std_thr=None):
        agent_names = list(agent_model_dict.keys())
        layers = list(agent_model_dict[agent_names[0]].best_model.state_dict().keys())

        std_model = deepcopy(agent_model_dict[agent_names[0]])
        state_dict_std = deepcopy(agent_model_dict[agent_names[0]].best_model.state_dict())
        if len(agent_model_dict.keys()) == 1:
            for layer in layers:
                state_dict_std.update({layer: agent_model_dict[agent_names[0]].model.state_dict()[layer]*0})
        else:
            state_dict_layer_mean = None
            for layer in layers:
                state_dict_layer_list = [agent_model_dict[agent_names[0]].best_model.state_dict()[layer]]
                dimension = len(agent_model_dict[agent_names[0]].best_model.state_dict()[layer].size())
                for agent in agent_names[1:]:
                    state_dict_layer_list.append(agent_model_dict[agent].best_model.state_dict()[layer])
                if dimension == 1:
                    state_dict_layer_std = torch.std(torch.stack(state_dict_layer_list, axis=1), axis=1) + EPSILON
                    if std_thr == 'auto':
                        state_dict_layer_mean = torch.mean(torch.stack(state_dict_layer_list, axis=1), axis=1)
                elif dimension == 2:
                    state_dict_layer_std = torch.std(torch.stack(state_dict_layer_list, axis=2), axis=2)
                    if std_thr == 'auto':
                        state_dict_layer_mean = torch.mean(torch.stack(state_dict_layer_list, axis=2), axis=2)
                elif dimension == 4:
                    state_dict_layer_std = torch.std(torch.stack(state_dict_layer_list, axis=4), axis=4)
                    if std_thr == 'auto':
                        state_dict_layer_mean = torch.mean(torch.stack(state_dict_layer_list, axis=4), axis=4)
                elif dimension == 0:
                    state_dict_layer_std = torch.tensor([EPSILON])
                    state_dict_layer_mean = torch.tensor([EPSILON])
                else:
                    raise NotImplemented

                if std_thr is not None:
                    if std_thr == 'standard':
                        state_dict_layer_std /= (len(agent_names) ** .5)
                    elif std_thr == 'auto':
                        index_of_disp = (state_dict_layer_std**2)/(torch.abs(state_dict_layer_mean) + EPSILON)
                        median_thr = torch.median(index_of_disp)
                        state_dict_layer_std[index_of_disp > median_thr] = (median_thr/len(agent_names))
                    else:
                        state_dict_layer_std[state_dict_layer_std > std_thr] = std_thr

                state_dict_std.update({layer: state_dict_layer_std})
        std_model.best_model.state_dict().update(state_dict_std)
        std_model.best_model.load_state_dict(state_dict_std)
        std_model.model.state_dict().update(state_dict_std)
        std_model.model.load_state_dict(state_dict_std)
        return std_model

    @staticmethod
    def sample(average_model, std_model):
        layers = list(average_model.model.state_dict().keys())

        sample_model = deepcopy(average_model)

        state_dict_sample = deepcopy(sample_model.best_model.state_dict())
        for layer in layers:
            noise = std_model.model.state_dict()[layer] * torch.normal(0, 1, std_model.model.state_dict()[layer].shape)
            state_dict_layer_sample = average_model.model.state_dict()[layer] + noise
            state_dict_sample.update({layer: state_dict_layer_sample})

        sample_model.best_model.state_dict().update(state_dict_sample)
        sample_model.best_model.load_state_dict(state_dict_sample)
        sample_model.model.state_dict().update(state_dict_sample)
        sample_model.model.load_state_dict(state_dict_sample)
        return sample_model
