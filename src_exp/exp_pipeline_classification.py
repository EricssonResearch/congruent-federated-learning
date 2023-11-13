from src.boosting_pipeline_classification import *
from src.models_classifier import MLPClassifier

import os
import torch
import pickle
import numpy as np

EPSILON = 1e-20


class ExpPipelineClassification(object):
    def __init__(self, config, exp_name):
        self.exp_name = exp_name
        self.config = config
        self._test_ratio = config['test_ratio']
        self._data_path = config['data_path']
        self._tasks = config['tasks']
        self._temp_path = config['temp_path']
        self._save_dir = config['save_dir']

    def congruent_federated_training(self, init_net, config, save_name):
        eval_dict_list = []
        eval_dict_local_list = []
        init_net_ave, init_net_std, eval_dict, local_dict = \
            self._congruent_federated_training(init_net=deepcopy(init_net),
                                               init_net_std=None,
                                               config=config,
                                               save_name=save_name,
                                               double_stochasticity=True)
        eval_dict_list.append(eval_dict)
        eval_dict_local_list.append(local_dict)
        for r in range(1, config['n_rounds_fl']):
            init_net_ave, init_net_std, eval_dict, local_dict = \
                self._congruent_federated_training(init_net=deepcopy(init_net_ave),
                                                   init_net_std=deepcopy(init_net_std),
                                                   config=config,
                                                   save_name=save_name,
                                                   double_stochasticity=True)
            eval_dict_list.append(eval_dict)
            eval_dict_local_list.append(local_dict)

        for agent in config['agent_ids']:
            self.save([eval_dict_list[_][agent] for _ in range(len(eval_dict_list))],
                      save_dir=self._save_dir + 'agent_' + agent + '/' + self.exp_name + '/', name=save_name)
            self.save([eval_dict_local_list[_][agent] for _ in range(len(eval_dict_list))],
                      save_dir=self._save_dir + 'agent_' + agent + '/' + self.exp_name + '/', name=save_name + '_local')

    def _congruent_federated_training(self, init_net, init_net_std, config, save_name,
                                      double_stochasticity, mu=None):

        model_agent_dict = {}
        eval_dict_local = {}
        for agent in config['agent_ids']:
            bootstrapped_net, local_eval = self.congruent_training(config, target_agent=agent,
                                                                   init_net=init_net,
                                                                   init_net_std=init_net_std,
                                                                   save_name=save_name,
                                                                   out_random_sample=double_stochasticity,
                                                                   mu=mu)
            model_agent = deepcopy(bootstrapped_net)
            model_agent_dict.update({agent: model_agent})
            eval_dict_local.update({agent: local_eval})

        acf = AgentMultiRoundAveraging(
            local_trainer=model_agent_dict,
            agent_ids=config['agent_ids'],
            data_obj=SimpleRead(data_path=self._data_path, tasks=self._tasks),
            test_ratio=self._test_ratio,
            use_best_model=False,
            val_ratio=None,
            std_thr=0.,
            print_key='agent',
            if_print=True)

        acf.average_predict_evaluate(model_agent_dict)
        ave_kpi = [np.mean([acf.eval_dict[key][task]['acc'] for
                            key in config['agent_ids']]) for task in self._tasks]
        n_estimators = len(config['estimator_ids'])
        print(f'cfl-b{n_estimators}: ave_kpi={ave_kpi}')
        return acf.federated_trainer, acf.federated_trainer_std, acf.eval_dict, eval_dict_local

    def congruent_training(self, config, target_agent, init_net, init_net_std=None, save_name=None,
                           out_random_sample=False, mu=None):
        config.update({'estimator_ids': [str(_) for _ in range(2)]})
        config.update({'estimator_num_epoch': 10, 'estimator_n_rounds': 5})
        model_estimator_dict = {}
        for estimator in config['estimator_ids']:
            if init_net_std is None:
                init_net.update(np.random.randint(0, 1e16, 1))
                target_net = deepcopy(init_net)
            else:
                init_net_full = self.sample(average_model=init_net, std_model=init_net_std)
                target_net = deepcopy(init_net_full.model)

            model_estimator = MLPClassifier(model=deepcopy(target_net),
                                            num_epochs=config['num_epochs_fl'],
                                            learning_rate=self.config['learning_rate'],
                                            batch_size=self.config['batch_size'],
                                            use_best_model=False,
                                            min_num_epochs=50,
                                            save_path=self._temp_path,
                                            if_print=False
                                            )
            model_estimator_dict.update({estimator: model_estimator})

        acf = AgentMultiRoundProbConditionalFederatedAveraging(
            local_trainer=model_estimator_dict,
            n_rounds=config['estimator_n_rounds'],
            agent_ids=config['estimator_ids'],
            data_obj=EstimatorRead(data_path=self._data_path,
                                   tasks=self._tasks, agent_id=target_agent),
            test_ratio=self._test_ratio,
            val_ratio=None,
            num_epochs=config['estimator_num_epoch'],
            use_best_model=False,
            std_thr='auto',
            print_key='target_agent.' + target_agent + '__bootstrapping',
            if_print=False)
        if init_net_std is None:
            acf.fit_predict_evaluate_init(mu=mu)
        else:
            acf.fit_predict_evaluate(mu=mu)
        ave_kpi = [np.mean([acf.eval_dict_list[-1][key][task]['acc'] for
                            key in config['estimator_ids']]) for task in self._tasks]
        print(f'target_agent {target_agent}: ave_kpi={ave_kpi}')
        for estimator in config['estimator_ids']:
            self.save(acf.eval_dict_list,
                      save_dir=self._save_dir + 'estimator_' + estimator + '/' + self.exp_name + '/', name=save_name)
        if out_random_sample is True:
            return acf.federated_trainer_sample, acf.eval_dict_list[-1]['0']
        return acf.federated_trainer, acf.eval_dict_list[-1]['0']

    @staticmethod
    def save(result, save_dir, name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + name + '__kpi.pickle', 'wb') as handle:
            pickle.dump(result, handle)

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

    def federated_training(self, init_net, config, save_name):
        amf = AgentMultiRoundFederatedAveraging(local_trainer=MLPClassifier(model=deepcopy(init_net),
                                                                            num_epochs=config['num_epochs'],
                                                                            learning_rate=self.config['learning_rate'],
                                                                            batch_size=self.config['batch_size'],
                                                                            use_best_model=False,
                                                                            min_num_epochs=50,
                                                                            save_path=self._temp_path,
                                                                            if_print=False
                                                                            ),
                                                n_rounds=config['n_rounds'],
                                                agent_ids=config['agent_ids'],
                                                data_obj=SimpleRead(data_path=self._data_path, tasks=self._tasks),
                                                test_ratio=self._test_ratio,
                                                val_ratio=None,
                                                if_print=False)
        amf.fit_predict_evaluate()
        ave_kpi = [np.mean([amf.eval_dict_list[-1][key][task]['acc'] for
                            key in config['agent_ids']]) for task in self._tasks]
        print(f' fl: ave_kpi = {ave_kpi}')
        for target_agent in config['agent_ids']:
            self.save([amf.eval_dict_list[_][target_agent] for _ in range(len(amf.eval_dict_list))],
                      save_dir=self._save_dir + 'agent_' + target_agent + '/' + self.exp_name + '/',
                      name=save_name)
